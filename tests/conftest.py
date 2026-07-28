"""Shared test fixtures."""

from __future__ import annotations

import base64
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def temp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_audio_path(temp_dir):
    """Create a small valid WAV file for testing."""
    p = temp_dir / "test.wav"
    # Minimal WAV header + 1ms of silence (44 bytes header + data)
    sample_rate = 44100
    bits = 16
    channels = 1
    block_align = channels * bits // 8
    byte_rate = sample_rate * block_align
    data_size = 0  # no audio data
    header = (
        b"RIFF"
        + (36 + data_size).to_bytes(4, "little")
        + b"WAVE"
        + b"fmt "
        + (16).to_bytes(4, "little")
        + (1).to_bytes(2, "little")
        + channels.to_bytes(2, "little")
        + sample_rate.to_bytes(4, "little")
        + byte_rate.to_bytes(4, "little")
        + block_align.to_bytes(2, "little")
        + bits.to_bytes(2, "little")
        + b"data"
        + data_size.to_bytes(4, "little")
    )
    p.write_bytes(header)
    return p


@pytest.fixture
def sample_large_audio_path(temp_dir):
    """Create a WAV file larger than 1MB for size tests."""
    p = temp_dir / "large.wav"
    p.write_bytes(b"\x00" * (2 * 1024 * 1024))  # 2MB
    return p


@pytest.fixture
def sample_mp3_path(temp_dir):
    """Create a minimal MP3-like file (just a header stub)."""
    p = temp_dir / "song.mp3"
    # MP3 sync word + header bytes (not a valid MP3, but has the right extension)
    p.write_bytes(b"\xff\xfb\x90\x00" + b"\x00" * 100)
    return p


@pytest.fixture
def sample_audio_base64(sample_audio_path):
    raw = sample_audio_path.read_bytes()
    return base64.b64encode(raw).decode("ascii")
