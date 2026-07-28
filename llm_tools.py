"""LLM 工具业务逻辑。"""

from __future__ import annotations

try:
    from .audio_utils import AudioError, download_audio_file, extract_file_component, load_audio, load_audio_from_path
    from .gemini_client import GeminiClient, GeminiClientError
except ImportError:
    from audio_utils import AudioError, download_audio_file, extract_file_component, load_audio, load_audio_from_path
    from gemini_client import GeminiClient, GeminiClientError


async def run_audio_analysis(
    event,
    supported_formats: list[str],
    max_size_mb: int,
    client: GeminiClient,
) -> str:
    """从当前 event 的 File 组件执行音频分析。"""
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


async def run_audio_analysis_from_path(
    file_path: str,
    max_size_mb: int,
    client: GeminiClient,
) -> str:
    """从本地文件路径执行音频分析。"""
    try:
        audio = await load_audio_from_path(file_path, max_size_mb)
        result = await client.analyze(audio.b64, audio.mime_type)
    except (AudioError, GeminiClientError) as e:
        return f"音频分析失败：{e}"
    else:
        return result


async def resolve_audio_ref(
    item: dict,
    max_size_mb: int,
) -> str:
    """将缓存的音频引用解析为可用路径。local 直接用，remote 按需下载。"""
    ref = item["ref"]
    if not ref:
        raise AudioError("文件引用为空。")

    if item["is_local"]:
        from pathlib import Path
        if Path(ref).is_file():
            return ref
        raise AudioError(f"文件已过期或不可访问：{ref}")

    # 远程 URL，按需下载
    name = item.get("name", "audio")
    dest = await download_audio_file(ref, name)
    item["ref"] = str(dest)
    item["is_local"] = True
    return str(dest)
