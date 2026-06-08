"""LLM 工具业务逻辑，与 AstrBot 框架解耦。

main.py 中的 @llm_tool 方法只做参数校验和转发，
实际分析逻辑全部在这里，方便独立测试和复用。
"""

from __future__ import annotations

from .audio_utils import AudioError, extract_file_component, load_audio
from .gemini_client import GeminiClient, GeminiClientError


async def run_audio_analysis(
    event,
    supported_formats: list[str],
    max_size_mb: int,
    client: GeminiClient,
) -> str:
    """执行音频分析，返回适合作为 llm_tool 返回值的字符串。

    成功返回分析结果描述；失败返回错误说明（均可直接被 LLM 引用）。
    调用方负责 client 的生命周期（close）。
    """
    file_comp = extract_file_component(event)
    if file_comp is None:
        return "当前消息中没有找到音频文件，无法进行分析。"

    try:
        audio = await load_audio(file_comp, supported_formats, max_size_mb)
        result = await client.analyze(audio.b64, audio.mime_type)
    except (AudioError, GeminiClientError) as e:
        return f"音频分析失败：{e}"
    else:
        return f"音频分析结果：{result}"
