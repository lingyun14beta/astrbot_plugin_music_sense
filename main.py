"""astrbot_plugin_music_sense — 让 Bot 听懂音乐。

分析群里分享的音频文件，让 Bot 能就音乐展开自然对话。

  触发方式：
  - /分析音频 [序号] [追问] + 音频文件 或 引用音频消息
  - LLM 工具 list_audio_files / analyze_audio_by_number
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent, MessageChain
from astrbot.api.event import filter as astr_filter
from astrbot.api.event.filter import CustomFilter
from astrbot.api.star import Context, Star

from .audio_utils import (
    AudioError,
    extract_file_component,
    load_audio,
    resolve_component_ref,
)
from .gemini_client import GeminiClient, GeminiClientError
from .llm_tools import (
    resolve_audio_ref,
    run_audio_analysis,
    run_audio_analysis_from_path,
)

_DEFAULT_SYSTEM_PROMPT = (
    "像一个懂音乐的朋友随便聊聊这段音乐，突出你最想说的。不要列表。60字以内。"
)

_ERROR_PREFIXES = ("文件处理失败", "音频分析失败")


def _is_error(text: str) -> bool:
    return text.startswith(_ERROR_PREFIXES)


def _format_relative_time(ts: float) -> str:
    """将时间戳格式化为相对时间（刚刚/N分钟前/N小时前/N天前）。"""
    diff = time.time() - ts
    if diff < 60:
        return "刚刚"
    if diff < 3600:
        return f"{int(diff / 60)}分钟前"
    if diff < 86400:
        return f"{int(diff / 3600)}小时前"
    return f"{int(diff / 86400)}天前"


def _format_audio_list(items: list[dict]) -> str:
    """格式化缓存音频列表（含相对接收时间），供 LLM 分辨新旧。"""
    lines = []
    for i, item in enumerate(items):
        when = _format_relative_time(item.get("received_at", 0))
        tag = " [已分析]" if item["result"] else ""
        lines.append(f"{i + 1}. {item['name']}（{when}）{tag}")
    return "\n".join(lines)


class _FileComponentFilter(CustomFilter):
    """匹配含有 File 组件的消息（含引用消息），用于缓存音频元数据。"""

    def filter(self, event: AstrMessageEvent, cfg: AstrBotConfig) -> bool:
        for comp in getattr(event.message_obj, "message", []):
            if type(comp).__name__ == "File":
                return True
            if type(comp).__name__ == "Reply":
                for rc in getattr(comp, "chain", []) or []:
                    if type(rc).__name__ == "File":
                        return True
        return False


class MusicSensePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
        self._registry: dict[str, list[dict]] = {}
        self._pending_injections: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._auto_sem = asyncio.Semaphore(3)  # 最多 3 个并发自动分析
        self._bg_tasks: set[asyncio.Task] = set()
        self._audio_hints: dict[str, set[str]] = {}  # umo -> 已提示过 LLM 的音频名
        self._pending_hints: dict[str, list[str]] = {}  # umo -> 待注入的音频名
        logger.info("[MusicSense] 插件已加载，支持格式：%s", self._supported_formats)

    # ------------------------------------------------------------------
    # 配置快捷属性
    # ------------------------------------------------------------------

    @property
    def _analysis_cfg(self) -> dict:
        return self.config.get("analysis", {})

    @property
    def _supported_formats(self) -> list[str]:
        return self._analysis_cfg.get(
            "supported_formats",
            ["mp3", "wav", "flac", "m4a", "aac", "ogg"],
        )

    @property
    def _max_size_mb(self) -> int:
        return int(self._analysis_cfg.get("max_file_size_mb", 20))

    @property
    def _max_cached_files(self) -> int:
        return max(1, int(self._analysis_cfg.get("max_cached_files", 50)))

    @property
    def _system_prompt(self) -> str:
        return self._analysis_cfg.get("system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()

    @property
    def _auto_analyze(self) -> bool:
        return bool(self._analysis_cfg.get("auto_analyze", False))

    @property
    def _inject_context(self) -> bool:
        return bool(self._analysis_cfg.get("inject_context", False))

    @property
    def _separate_prompts(self) -> bool:
        return bool(self._analysis_cfg.get("separate_prompts", False))

    @property
    def _debug(self) -> bool:
        return bool(self._analysis_cfg.get("debug", False))

    @property
    def _command_system_prompt(self) -> str:
        return self._analysis_cfg.get(
            "command_system_prompt", _DEFAULT_SYSTEM_PROMPT
        ).strip()

    def _resolve_provider_and_model(self) -> tuple[dict, str]:
        providers: list = self.config.get("api_provider", [])
        if not isinstance(providers, list):
            providers = []

        fallback_provider = providers[0] if providers else {}
        fallback_model = fallback_provider.get("model", "gemini-2.0-flash")

        model_cfg: str = self._analysis_cfg.get("model", "").strip()
        if not model_cfg or "/" not in model_cfg:
            return fallback_provider, fallback_model

        provider_name, model_name = model_cfg.split("/", 1)
        provider_name = provider_name.strip()
        model_name = model_name.strip()

        if not provider_name or not model_name:
            return fallback_provider, fallback_model

        for p in providers:
            if isinstance(p, dict) and p.get("name", "").strip() == provider_name:
                return p, model_name or p.get("model", "gemini-2.0-flash")

        return fallback_provider, fallback_model

    def _make_client(
        self, extra_prompt: str = "", use_command_prompt: bool = False
    ) -> GeminiClient:
        provider, model = self._resolve_provider_and_model()
        if self._separate_prompts and use_command_prompt:
            sp = self._command_system_prompt
        else:
            sp = self._system_prompt
        if extra_prompt:
            sp = f"{sp}\n\n用户的追加问题：{extra_prompt}"
        return GeminiClient(
            api_key=provider.get("api_key", ""),
            model=model,
            system_prompt=sp,
            base_url=provider.get("base_url", ""),
            timeout=int(provider.get("timeout", 120)),
        )

    # ------------------------------------------------------------------
    # 音频文件缓存钩子
    # ------------------------------------------------------------------

    @astr_filter.custom_filter(_FileComponentFilter)
    async def _on_file_message(self, event: AstrMessageEvent):
        """消息中包含 File 组件时，缓存其元数据。不下载，不下发消息。"""
        umo = event.unified_msg_origin

        def _cache(comp, items):
            name = getattr(comp, "name", "") or ""
            ext = Path(name).suffix.lstrip(".").lower()
            if ext not in self._supported_formats:
                if self._debug:
                    logger.info(
                        "[MusicSense] 跳过非音频文件：%s (扩展名 .%s)",
                        name or "(无名称)",
                        ext,
                    )
                return
            local, url = resolve_component_ref(comp)
            ref = local if (local and Path(local).is_file()) else (url or local)
            is_local = bool(local and Path(local).is_file())
            # 去重：同名同引用的音频（如反复引用同一消息）只缓存一次，但刷新接收时间
            for it in items:
                if it["name"] == name and it["ref"] == ref:
                    it["received_at"] = time.time()
                    return
            items.append(
                {
                    "name": name,
                    "ref": ref,
                    "is_local": is_local,
                    "result": None,
                    "received_at": time.time(),
                }
            )

        new_items = []
        new_hints = []
        async with self._lock:
            items = self._registry.setdefault(umo, [])
            hinted = self._audio_hints.setdefault(umo, set())
            for comp in getattr(event.message_obj, "message", []):
                if type(comp).__name__ == "File":
                    before = len(items)
                    _cache(comp, items)
                    if len(items) > before:
                        new_items.append(items[-1])
                elif type(comp).__name__ == "Reply":
                    for rc in getattr(comp, "chain", []) or []:
                        if type(rc).__name__ == "File":
                            before = len(items)
                            _cache(rc, items)
                            if len(items) > before:
                                new_items.append(items[-1])
            # 记录需要提示 LLM 的新音频（未提示过的），仅在 LLM 工具启用时注入
            if self._analysis_cfg.get("enable_llm_tool", True):
                for item in items:
                    if item["name"] not in hinted:
                        new_hints.append(item["name"])
                        hinted.add(item["name"])
                if new_hints:
                    self._pending_hints.setdefault(umo, []).extend(new_hints)

        if self._auto_analyze:
            triggered = 0
            for item in new_items:
                if item["is_local"]:
                    if self._debug:
                        logger.info("[MusicSense] 触发自动分析：%s", item.get("name"))
                    asyncio.create_task(self._auto_analyze_task(umo, item))
                    triggered += 1
            if self._debug and new_items and not triggered:
                logger.info("[MusicSense] 所有新文件均为远程，跳过自动分析")

        if self._debug and new_items:
            logger.info("[MusicSense] 缓存完成：%d 个音频文件", len(new_items))

        # 缓存上限裁剪：保留最近的 N 个，避免长会话无限膨胀
        max_cached = self._max_cached_files
        if len(items) > max_cached:
            overflow = len(items) - max_cached
            async with self._lock:
                del items[:overflow]
            if self._debug:
                logger.info("[MusicSense] 缓存超限，裁剪 %d 个最旧条目", overflow)

        yield

    async def _auto_analyze_task(self, umo: str, item: dict) -> None:
        """后台自动分析音频，可选注入对话上下文。"""
        async with self._auto_sem:
            async with self._lock:
                if item["result"]:
                    return
            try:
                resolved = await resolve_audio_ref(item, self._max_size_mb)
            except AudioError:
                logger.warning("[MusicSense] 自动分析：文件不可用 %s", item.get("name"))
                return

            client = self._make_client()
            try:
                result = await run_audio_analysis_from_path(
                    resolved, self._max_size_mb, client
                )
            except Exception:
                logger.warning(
                    "[MusicSense] 自动分析失败：%s", item.get("name"), exc_info=True
                )
                return
            finally:
                await client.close()

            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
                if self._debug:
                    logger.info("[MusicSense] 自动分析完成：%s", item.get("name"))
                if self._inject_context:
                    async with self._lock:
                        self._pending_injections.setdefault(umo, []).append(
                            {
                                "role": "user",
                                "content": f"[音乐感知] 刚刚收到的音频「{item['name']}」分析：{result}",
                            }
                        )

    @astr_filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """每次 LLM 请求前注入待发送的分析结果与"对话中有音频"提示。

        「音频感知」提示每音频仅注入一次（不消耗 API），引导 LLM 调用分析工具：
        LLM 上下文中音频消息只是占位符，它不知道音频可分析。
        """
        umo = event.unified_msg_origin
        async with self._lock:
            pending_hints = self._pending_hints.pop(umo, [])
            pending = self._pending_injections.pop(umo, [])

        if pending_hints:
            names = "、".join(pending_hints[:5])
            req.contexts.append(
                {
                    "role": "user",
                    "content": (
                        f"[音频感知] 本对话中有音频文件：{names}。"
                        "如果用户提到音乐、音频或想了解其内容，"
                        "请调用 list_audio_files 或 analyze_audio_by_number 工具分析音频。"
                    ),
                }
            )
            if self._debug:
                logger.info("[MusicSense] 已注入音频感知提示：%s", names)

        if pending:
            if self._debug:
                logger.info("[MusicSense] 注入 %d 条分析到上下文", len(pending))
            req.contexts.extend(pending)

    # ------------------------------------------------------------------
    # 后台分析（LLM 工具场景：立即返回，结果唤醒 AI 发送）
    # ------------------------------------------------------------------

    def _run_background_analysis(self, umo: str, coro) -> None:
        """后台执行分析协程，完成后唤醒主 Agent 以角色口吻发送结果。

        AstrBot 对 LLM 工具调用有 120 秒硬超时，音频分析可能超时，
        故工具只提交任务立即返回，耗时分析在此后台执行（无超时限制）。
        """

        async def _wrapper() -> None:
            try:
                result = await coro
            except Exception as e:
                logger.error("[MusicSense] 后台分析异常", exc_info=True)
                await self._deliver_result(umo, f"音频分析失败：{e}")
                return
            await self._deliver_result(umo, result)

        task = asyncio.create_task(_wrapper())
        self._bg_tasks.add(task)
        task.add_done_callback(self._bg_tasks.discard)

    async def _deliver_result(self, umo: str, result_text: str) -> None:
        """交付分析结果：优先唤醒 AI 以角色口吻发送，失败兜底直接发送。"""
        try:
            sent = await self._wake_ai_for_result(umo, result_text)
        except Exception as e:
            logger.warning("[MusicSense] 唤醒 AI 发送失败，改为直接发送：%s", e)
            sent = False
        if not sent:
            await self._safe_send(umo, result_text)

    async def _wake_ai_for_result(self, umo: str, result_text: str) -> bool:
        """唤醒主 Agent 处理音频分析结果（借鉴 image_generation 的任务完成唤醒机制）。

        构造一次带完整会话上下文的主动 Agent 回合，让 LLM 用角色口吻
        通过 send_message_to_user 把结果发送给用户。

        Returns:
            AI 是否成功发送了消息。
        """
        from astrbot.core.agent.tool import ToolSet
        from astrbot.core.astr_main_agent import (
            MainAgentBuildConfig,
            _get_session_conv,
            build_main_agent,
        )
        from astrbot.core.cron.events import CronMessageEvent
        from astrbot.core.platform.message_session import MessageSession
        from astrbot.core.provider.entities import ProviderRequest
        from astrbot.core.tools.message_tools import SendMessageToUserTool

        system_prompt = (
            "你是一个自主 Agent。你被唤醒是因为之前提交的音频分析任务已完成。\n"
            "# 重要规则\n"
            "1. 这不是普通对话回合，不要寒暄，不要提问。\n"
            "2. 你必须使用 send_message_to_user 工具把分析结果发送给用户，否则用户看不到。\n"
            "3. 用你平时的角色口吻组织语言（保持人设），但不要改变事实内容。\n"
            "4. 如果任务失败，用角色口吻简要说明失败原因。\n"
            "# 音频分析结果\n"
            f"{result_text}"
        )

        session = MessageSession.from_str(umo)
        cron_event = CronMessageEvent(
            context=self.context,
            session=session,
            message=f"音频分析任务完成：{result_text[:100]}",
            sender_id="astrbot",
            sender_name="MusicSense",
            message_type=session.message_type,
        )

        cfg = self.context.get_config(umo=umo)
        provider_settings = cfg.get("provider_settings", {})
        tool_call_timeout = provider_settings.get("tool_call_timeout", 120)
        provider = self.context.get_using_provider(umo)

        req = ProviderRequest()
        req.conversation = await _get_session_conv(
            event=cron_event,
            plugin_context=self.context,
        )
        history_context = json.loads(req.conversation.history or "[]")
        if history_context:
            req.contexts = history_context
            context_dump = req._print_friendly_context()
            req.contexts = []
            req.system_prompt += (
                f"\n\n以下是你和用户之前的对话历史：\n---\n{context_dump}\n---\n"
            )
        req.system_prompt += system_prompt
        req.prompt = "请按系统指令把音频分析结果发送给用户。"
        req.func_tool = ToolSet()
        req.func_tool.add_tool(
            self.context.get_llm_tool_manager().get_builtin_tool(SendMessageToUserTool)
        )

        config = MainAgentBuildConfig(
            tool_call_timeout=tool_call_timeout,
            llm_safety_mode=False,
            streaming_response=False,
            provider_settings=provider_settings,
            computer_use_runtime="none",
            add_cron_tools=False,
        )
        result = await build_main_agent(
            event=cron_event,
            plugin_context=self.context,
            config=config,
            provider=provider,
            req=req,
            apply_reset=False,
        )
        if not result:
            return False

        # 裁剪工具：本次主动回合只允许发送消息
        result.provider_request.func_tool = ToolSet()
        result.provider_request.func_tool.add_tool(
            self.context.get_llm_tool_manager().get_builtin_tool(SendMessageToUserTool)
        )
        if result.reset_coro:
            await result.reset_coro

        sent = False
        runner = result.agent_runner
        async for agent_resp in runner.step_until_done(30):
            if agent_resp.type != "tool_call_result":
                continue
            chain = agent_resp.data.get("chain")
            if not chain:
                continue
            content = chain.get_plain_text(with_other_comps_mark=True)
            if "Message sent to session" in content:
                sent = True
                break
        return sent

    async def _safe_send(self, umo: str, text: str) -> None:
        """向会话主动发送文本，失败仅记日志（不中断后台任务）。"""
        try:
            await self.context.send_message(umo, MessageChain().message(text))
        except Exception:
            logger.warning("[MusicSense] 主动发送分析结果失败", exc_info=True)

    # ------------------------------------------------------------------
    # 指令处理
    # ------------------------------------------------------------------

    @astr_filter.command("分析音频")
    async def handle_music_command(self, event: AstrMessageEvent):
        """分析音频文件。用法：/分析音频 [序号|追问] [音频文件]"""
        file_comp = extract_file_component(event)
        extra = _extract_extra(event.message_str, "分析音频")

        if file_comp is not None:
            yield event.plain_result("正在分析音乐，请稍候...")
            result = await self._analyze_file_comp(
                file_comp, extra, use_command_prompt=True
            )
            yield event.plain_result(result)
            # 同步缓存结果（排除错误）
            if not _is_error(result):
                name = getattr(file_comp, "name", "") or ""
                async with self._lock:
                    for item in self._registry.get(event.unified_msg_origin, []):
                        if item["name"] == name and item["result"] is None:
                            item["result"] = result
                            break
            return

        # 当前消息和引用消息中都没有 File 组件，尝试从缓存取
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            yield event.plain_result(
                "没有找到音频文件，请在发送命令时同时附带音频文件，"
                "或引用一条含音频文件的消息。",
            )
            return

        # 解析序号和追问："/分析音频 1 这首歌好听吗" → idx=0, question="这首歌好听吗"
        idx = -1
        question = ""
        if extra:
            parts = extra.split(None, 1)
            try:
                idx = int(parts[0]) - 1
                question = parts[1] if len(parts) > 1 else ""
            except (ValueError, TypeError):
                question = extra  # 不是数字，全部当追问，但没指定序号

        if idx < 0:
            # 没有指定序号，列出缓存让用户选
            yield event.plain_result(
                "对话中有以下音频文件，请指定序号，如：/分析音频 1\n"
                + _format_audio_list(items)
            )
            return

        if idx >= len(items):
            yield event.plain_result(f"序号无效，可选范围 1-{len(items)}。")
            return

        item = items[idx]
        if item["result"]:
            yield event.plain_result(f"「{item['name']}」(已缓存) {item['result']}")
            return

        try:
            resolved = await resolve_audio_ref(item, self._max_size_mb)
        except AudioError as e:
            yield event.plain_result(f"「{item['name']}」文件不可用：{e}")
            return

        yield event.plain_result(f"正在分析「{item['name']}」，请稍候...")
        client = self._make_client(question, use_command_prompt=True)
        try:
            result = await run_audio_analysis_from_path(
                resolved, self._max_size_mb, client
            )
            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
            yield event.plain_result(f"「{item['name']}」{result}")
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # LLM 工具
    # ------------------------------------------------------------------

    @llm_tool("analyze_current_audio")
    async def analyze_current_audio(self, event: AstrMessageEvent, question: str = ""):
        """分析本条消息中直接携带的音频文件（用户刚刚随消息发送的，或引用消息中的音频）。
        仅当本条消息（或引用的消息）确实带有音频时调用本工具。
        如果用户提到的音频不是本条消息附带的（是之前发过的），不要调用本工具，
        请先调用 list_audio_files 查看对话中的音频列表，再用 analyze_audio_by_number 按序号分析。

        Args:
            question(str, optional): 用户对音频的具体追问，如"这首歌好听吗？"
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "音频分析功能未启用。"

        umo = event.unified_msg_origin
        if extract_file_component(event) is None:
            # 当前消息没有携带音频：列出缓存（含时间），引导 LLM 按语境选序号
            async with self._lock:
                items = list(self._registry.get(umo, []))
            if not items:
                return (
                    "当前消息中没有音频文件，对话中也没有缓存音频，请让用户先发送音频。"
                )
            return (
                "当前消息没有附带音频。对话中的音频文件（按接收时间）：\n"
                + _format_audio_list(items)
                + "\n请根据用户语境判断指的是哪个，"
                "调用 analyze_audio_by_number 指定序号分析。"
            )

        self._run_background_analysis(umo, self._analyze_current_async(event, question))
        return (
            "⏳ 音频分析任务已提交，正在后台执行（音频分析可能需要数十秒）。"
            "分析完成后插件会直接把结果发送给用户，"
            "无需重复调用分析工具，等待结果即可。"
        )

    async def _analyze_current_async(
        self, event: AstrMessageEvent, question: str
    ) -> str:
        """后台执行当前消息音频分析，返回要发送给用户的结果文本。"""
        client = self._make_client(question)
        try:
            result = await run_audio_analysis(
                event,
                self._supported_formats,
                self._max_size_mb,
                client,
            )
        finally:
            await client.close()

        if not _is_error(result):
            file_comp = extract_file_component(event)
            if file_comp:
                name = getattr(file_comp, "name", "") or ""
                async with self._lock:
                    for item in self._registry.get(event.unified_msg_origin, []):
                        if item["name"] == name and item["result"] is None:
                            item["result"] = result
                            break
        return result

    @llm_tool("list_audio_files")
    async def list_audio_files(self, event: AstrMessageEvent):
        """列出当前对话中出现过的所有音频文件及其序号。
        当用户提到之前发过的音频、询问对话中有哪些音频，
        或需要按序号分析历史音频时调用。调用后再用 analyze_audio_by_number 分析指定音频。
        """
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "对话中未收到过音频文件。"

        return "对话中的音频文件（按接收时间）：\n" + _format_audio_list(items)

    @llm_tool("analyze_audio_by_number")
    async def analyze_audio_by_number(self, event: AstrMessageEvent, number: int):
        """分析对话中指定序号的音频文件（适用于之前发过、本条消息未附带的音频）。
        先调用 list_audio_files 获取序号，再调用本工具。

        Args:
            number(int): 音频文件序号，从 list_audio_files 返回的列表中选择
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "音频分析功能未启用。"

        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "未找到音频文件，请先发送音频文件。"
        try:
            number = int(number)
        except (TypeError, ValueError):
            return "序号无效，必须是整数。"
        if number < 1 or number > len(items):
            return f"序号无效，可选范围 1-{len(items)}。"

        item = items[number - 1]

        if item["result"]:
            return f"「{item['name']}」(已缓存结果) {item['result']}"

        self._run_background_analysis(umo, self._analyze_number_async(umo, item))
        return (
            f"⏳ 已提交后台分析「{item['name']}」（音频分析可能需要数十秒）。"
            "分析完成后插件会直接把结果发送给用户，"
            "无需重复调用分析工具，等待结果即可。"
        )

    async def _analyze_number_async(self, umo: str, item: dict) -> str:
        """后台执行按序号分析，返回要发送给用户的结果文本。"""
        try:
            resolved_path = await resolve_audio_ref(item, self._max_size_mb)
        except AudioError as e:
            return f"「{item['name']}」文件不可用：{e}"

        client = self._make_client()
        try:
            result = await run_audio_analysis_from_path(
                resolved_path, self._max_size_mb, client
            )
        finally:
            await client.close()

        if not _is_error(result):
            async with self._lock:
                item["result"] = result
        return f"「{item['name']}」{result}"

    # ------------------------------------------------------------------
    # 核心分析逻辑
    # ------------------------------------------------------------------

    async def _analyze_file_comp(
        self, file_comp, extra_prompt: str = "", use_command_prompt: bool = False
    ) -> str:
        client = self._make_client(extra_prompt, use_command_prompt)
        try:
            audio = await load_audio(
                file_comp,
                self._supported_formats,
                self._max_size_mb,
            )
            result = await client.analyze(audio.b64, audio.mime_type)
        except AudioError as e:
            return f"文件处理失败：{e}"
        except GeminiClientError as e:
            return f"音频分析失败：{e}"
        else:
            return result
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def terminate(self) -> None:
        for task in list(self._bg_tasks):
            task.cancel()
        self._bg_tasks.clear()
        self._registry.clear()
        self._pending_injections.clear()
        self._audio_hints.clear()
        self._pending_hints.clear()
        logger.info("[MusicSense] 插件已卸载")


def _extract_extra(message_str: str, command: str) -> str:
    if not message_str:
        return ""
    text = message_str.strip()
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix) :].strip()
    return ""
