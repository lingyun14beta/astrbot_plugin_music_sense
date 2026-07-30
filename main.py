"""astrbot_plugin_music_sense — 让 Bot 听懂音乐。

通过 Gemini 原生多模态音频理解，分析群里分享的音频文件，
将结果注入对话历史，让 Bot 能就音乐展开自然对话。

 触发方式：
  - /分析音频 [序号] [追问] + 音频文件 或 引用音频消息
  - LLM 工具 list_audio_files / analyze_audio_by_number
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter as astr_filter
from astrbot.api.star import Context, Star, register
from astrbot.api.event.filter import CustomFilter

from .audio_utils import AudioError, extract_file_component, load_audio, resolve_component_ref
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


@register(
    "astrbot_plugin_music_sense",
    "让 Bot 听懂音乐",
    "通过 Gemini 原生多模态音频理解分析音频文件，将结果注入对话历史。",
    "0.2.0",
)
class MusicSensePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
        self._registry: dict[str, list[dict]] = {}
        self._pending_injections: dict[str, list[dict]] = {}
        self._lock = asyncio.Lock()
        self._auto_sem = asyncio.Semaphore(3)  # 最多 3 个并发自动分析
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
        return self._analysis_cfg.get("command_system_prompt", _DEFAULT_SYSTEM_PROMPT).strip()

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

    def _make_client(self, extra_prompt: str = "", use_command_prompt: bool = False) -> GeminiClient:
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
                return
            local, url = resolve_component_ref(comp)
            if local and Path(local).is_file():
                items.append({"name": name, "ref": local, "is_local": True, "result": None})
            else:
                items.append({"name": name, "ref": url or local, "is_local": False, "result": None})

        new_items = []
        async with self._lock:
            items = self._registry.setdefault(umo, [])
            for comp in getattr(event.message_obj, "message", []):
                if type(comp).__name__ == "File":
                    _cache(comp, items)
                    new_items.append(items[-1])
                elif type(comp).__name__ == "Reply":
                    for rc in getattr(comp, "chain", []) or []:
                        if type(rc).__name__ == "File":
                            _cache(rc, items)
                            new_items.append(items[-1])

        if self._auto_analyze:
            for item in new_items:
                if item["is_local"]:
                    if self._debug:
                        logger.info("[MusicSense] 触发自动分析：%s", item.get("name"))
                    asyncio.create_task(self._auto_analyze_task(umo, item))

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
                result = await run_audio_analysis_from_path(resolved, self._max_size_mb, client)
            except Exception:
                logger.warning("[MusicSense] 自动分析失败：%s", item.get("name"), exc_info=True)
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
                        self._pending_injections.setdefault(umo, []).append({
                            "role": "user",
                            "content": f"[音乐感知] 刚刚收到的音频「{item['name']}」分析：{result}",
                        })

    @astr_filter.on_llm_request()
    async def _on_llm_request(self, event: AstrMessageEvent, req):
        """每次 LLM 请求前注入待发送的分析结果。"""
        if not self._inject_context:
            return
        async with self._lock:
            pending = self._pending_injections.pop(event.unified_msg_origin, [])
        if pending:
            if self._debug:
                logger.info("[MusicSense] 注入 %d 条分析到上下文", len(pending))
            req.contexts.extend(pending)

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
            result = await self._analyze_file_comp(file_comp, extra, use_command_prompt=True)
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
            lines = [f"{i + 1}. {it['name']}" for i, it in enumerate(items)]
            yield event.plain_result(
                "对话中有以下音频文件，请指定序号，如：/分析音频 1\n" + "\n".join(lines)
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
            result = await run_audio_analysis_from_path(resolved, self._max_size_mb, client)
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
    async def analyze_current_audio(self, event: AstrMessageEvent):
        """分析当前消息或引用消息中的音频文件。
        当用户直接发送了音频并希望 bot 理解、评价时调用。
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return "音频分析功能未启用。"

        client = self._make_client()
        try:
            result = await run_audio_analysis(
                event,
                self._supported_formats,
                self._max_size_mb,
                client,
            )
        finally:
            await client.close()

        file_comp = extract_file_component(event)
        if file_comp and not _is_error(result):
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
        当用户提到之前的音频但未指定具体哪个时调用。
        """
        umo = event.unified_msg_origin
        async with self._lock:
            items = list(self._registry.get(umo, []))

        if not items:
            return "对话中未收到过音频文件。"

        lines = []
        for i, item in enumerate(items):
            tag = " [已分析]" if item["result"] else ""
            lines.append(f"{i + 1}. {item['name']}{tag}")
        return "对话中的音频文件：\n" + "\n".join(lines)

    @llm_tool("analyze_audio_by_number")
    async def analyze_audio_by_number(self, event: AstrMessageEvent, number: int):
        """分析对话中指定序号的音频文件。需先调用 list_audio_files 获取序号。

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
        if number < 1 or number > len(items):
            return f"序号无效，可选范围 1-{len(items)}。"

        item = items[number - 1]

        if item["result"]:
            return f"「{item['name']}」(已缓存结果) {item['result']}"

        try:
            resolved_path = await resolve_audio_ref(item, self._max_size_mb)
        except AudioError as e:
            return f"「{item['name']}」文件不可用：{e}"

        client = self._make_client()
        try:
            result = await run_audio_analysis_from_path(resolved_path, self._max_size_mb, client)
            if not _is_error(result):
                async with self._lock:
                    item["result"] = result
            return f"「{item['name']}」{result}"
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # 核心分析逻辑
    # ------------------------------------------------------------------

    async def _analyze_file_comp(self, file_comp, extra_prompt: str = "", use_command_prompt: bool = False) -> str:
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
        self._registry.clear()
        self._pending_injections.clear()
        logger.info("[MusicSense] 插件已卸载")


def _extract_extra(message_str: str, command: str) -> str:
    if not message_str:
        return ""
    text = message_str.strip()
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return ""
