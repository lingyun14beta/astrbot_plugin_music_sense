"""音频文件工具：从消息链中提取 File 组件、校验格式与大小、读取为 base64。"""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent

# mime type 映射，Gemini 支持的音频格式
_MIME_MAP: dict[str, str] = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "flac": "audio/flac",
    "m4a": "audio/mp4",
    "aac": "audio/aac",
    "ogg": "audio/ogg",
    "opus": "audio/opus",
    "webm": "audio/webm",
}


@dataclass
class AudioFile:
    """已校验并读取完毕的音频文件。"""

    b64: str
    mime_type: str
    filename: str


class AudioError(Exception):
    """音频处理错误，message 可直接透传给 LLM。"""


def _get_ext(name: str) -> str:
    return Path(name).suffix.lstrip(".").lower()


def extract_file_component(event: AstrMessageEvent):
    """从消息链（含引用消息）中提取第一个 File 组件。

    优先检查当前消息链，找不到则检查 Reply 组件内的嵌套消息链。
    返回 File 组件实例，或 None。
    """
    messages = _get_messages(event)

    file_comp = _find_file_in_chain(messages)
    if file_comp is not None:
        return file_comp

    for comp in messages:
        if type(comp).__name__ == "Reply":
            chain = getattr(comp, "chain", None) or []
            file_comp = _find_file_in_chain(chain)
            if file_comp is not None:
                return file_comp

    return None


def _find_file_in_chain(chain) -> object | None:
    for comp in chain or []:
        if type(comp).__name__ == "File":
            return comp
    return None


def _get_messages(event: AstrMessageEvent) -> list:
    if hasattr(event, "get_messages"):
        try:
            msgs = event.get_messages()
            if msgs is not None:
                return list(msgs)
        except Exception:
            pass
    if hasattr(event, "message_obj") and hasattr(event.message_obj, "message"):
        return list(event.message_obj.message or [])
    return []


async def load_audio(
    file_comp,
    supported_formats: list[str],
    max_size_mb: int,
) -> AudioFile:
    """校验并读取 File 组件为 AudioFile。

    Args:
        file_comp: AstrBot File 消息组件实例。
        supported_formats: 允许的扩展名列表，如 ['mp3', 'wav']。
        max_size_mb: 文件大小上限（MB）。

    Raises:
        AudioError: 格式不支持、文件过大、读取失败等，message 可透传给 LLM。
    """
    name: str = getattr(file_comp, "name", "") or ""
    ext = _get_ext(name)

    if ext not in supported_formats:
        supported_str = "、".join(supported_formats)
        raise AudioError(
            f"文件格式 .{ext} 不支持，当前支持的格式：{supported_str}。",
        )

    mime_type = _MIME_MAP.get(ext, f"audio/{ext}")

    try:
        local_path: str = await file_comp.get_file()
    except Exception as e:
        raise AudioError(f"获取文件失败：{e}") from e

    p = Path(local_path) if local_path else None
    if not p or not p.is_file():
        raise AudioError("文件不存在或无法访问，请确认文件已上传完成。")

    size_bytes = p.stat().st_size
    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = size_bytes / 1024 / 1024
        raise AudioError(
            f"文件大小 {size_mb:.1f} MB 超过限制 {max_size_mb} MB，请上传较小的文件。",
        )

    try:
        raw = await asyncio.to_thread(p.read_bytes)
    except OSError as e:
        raise AudioError(f"读取文件失败：{e}") from e

    b64 = base64.b64encode(raw).decode("ascii")
    return AudioFile(b64=b64, mime_type=mime_type, filename=name)
