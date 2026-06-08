"""astrbot_plugin_music_sense — 让 Bot 听懂音乐。

通过 Gemini 原生多模态音频理解，分析群里分享的音频文件，
将结果注入对话历史，让 Bot 能就音乐展开自然对话。

触发方式：
  - 唤醒词 / @bot + 注册指令（支持 AstrBot 别名）+ 附带或引用音频文件
  - LLM 工具自动调用（需在配置中启用）
"""

from __future__ import annotations

from astrbot.api import AstrBotConfig, llm_tool, logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.event import filter as astr_filter
from astrbot.api.star import Context, Star, register

from .audio_utils import AudioError, extract_file_component, load_audio
from .gemini_client import GeminiClient, GeminiClientError
from .llm_tools import run_audio_analysis

_DEFAULT_SYSTEM_PROMPT = (
    "你是一个懂音乐的朋友，请用自然、生动的语言描述这段音乐的情感基调、"
    "氛围、风格和节奏感。不要用列表，像聊天一样说话，100字以内。"
)


@register(
    "astrbot_plugin_music_sense",
    "让 Bot 听懂音乐",
    "通过 Gemini 原生多模态音频理解分析音频文件，将结果注入对话历史。",
    "0.1.0",
)
class MusicSensePlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig = None) -> None:
        super().__init__(context)
        self.config: AstrBotConfig = config or {}
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

    def _resolve_provider_and_model(self) -> tuple[dict, str]:
        """根据 analysis.model（格式：供应商名称/模型名称）解析接入方和模型。

        解析失败或留空时，fallback 到第一个接入方，模型使用 gemini-2.0-flash。
        """
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
                resolved_model = model_name or p.get("model", "gemini-2.0-flash")
                return p, resolved_model

        return fallback_provider, fallback_model

    def _make_client(self, extra_prompt: str = "") -> GeminiClient:
        provider, model = self._resolve_provider_and_model()
        system_prompt = self._system_prompt
        if extra_prompt:
            system_prompt = f"{system_prompt}\n\n用户的追加问题：{extra_prompt}"
        return GeminiClient(
            api_key=provider.get("api_key", ""),
            model=model,
            system_prompt=system_prompt,
            base_url=provider.get("base_url", ""),
            timeout=int(provider.get("timeout", 120)),
        )

    # ------------------------------------------------------------------
    # 指令处理
    # ------------------------------------------------------------------

    @astr_filter.command("分析音频")
    async def handle_music_command(self, event: AstrMessageEvent):
        """分析音频文件的情感、氛围与风格。用法：/分析音频 [追问] [音频文件]"""
        file_comp = extract_file_component(event)
        if file_comp is None:
            yield event.plain_result(
                "没有找到音频文件，请在发送命令时同时附带音频文件，"
                "或引用一条含音频文件的消息。",
            )
            return

        # 提取命令词后面的追问文本
        extra = _extract_extra(event.message_str, "分析音频")

        yield event.plain_result("正在分析音乐，请稍候...")
        result = await self._analyze_file_comp(file_comp, extra)
        yield event.plain_result(result)

    # ------------------------------------------------------------------
    # LLM 工具
    # ------------------------------------------------------------------

    @llm_tool("analyze_current_audio")
    async def analyze_current_audio(self, event: AstrMessageEvent):
        """分析当前消息或引用消息中的音频文件，描述其情感基调、氛围、风格和节奏感。
        当用户发送了音频文件并希望 bot 理解、评价或讨论这段音乐时调用此工具。
        """
        if not self._analysis_cfg.get("enable_llm_tool", True):
            return None

        client = self._make_client()
        try:
            return await run_audio_analysis(
                event,
                self._supported_formats,
                self._max_size_mb,
                client,
            )
        finally:
            await client.close()

    # ------------------------------------------------------------------
    # 核心分析逻辑
    # ------------------------------------------------------------------

    async def _analyze_file_comp(self, file_comp, extra_prompt: str = "") -> str:
        """分析 File 组件，返回可直接展示给用户的文本。"""
        client = self._make_client(extra_prompt)
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
        logger.info("[MusicSense] 插件已卸载")


def _extract_extra(message_str: str, command: str) -> str:
    """从消息文本中提取命令词之后的追问内容。

    例如："/分析音频 这首歌适合什么场景" → "这首歌适合什么场景"
    """
    if not message_str:
        return ""
    text = message_str.strip()
    # 去掉可能的斜杠前缀
    for prefix in (f"/{command}", command):
        if text.startswith(prefix):
            return text[len(prefix):].strip()
    return ""
