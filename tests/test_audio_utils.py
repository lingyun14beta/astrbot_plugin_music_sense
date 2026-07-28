"""Tests for audio_utils.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from audio_utils import (
    AudioError,
    _get_ext,
    _read_and_validate,
    download_audio_file,
    load_audio_from_path,
    resolve_component_ref,
)


class TestGetExt:
    def test_simple_extension(self):
        assert _get_ext("song.mp3") == "mp3"

    def test_no_extension(self):
        assert _get_ext("song") == ""

    def test_multiple_dots(self):
        assert _get_ext("song.2024.mp3") == "mp3"

    def test_uppercase(self):
        assert _get_ext("SONG.MP3") == "mp3"

    def test_path_with_dirs(self):
        assert _get_ext("/home/user/music/song.wav") == "wav"


class TestResolveComponentRef:
    def test_local_file_path(self):
        class MockComp:
            file_ = r"C:\Users\test\song.mp3"
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == r"C:\Users\test\song.mp3"
        assert url == ""

    def test_file_uri(self):
        class MockComp:
            file_ = "file:///C:/Users/test/song.mp3"
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == "C:/Users/test/song.mp3"

    def test_remote_url_only(self):
        class MockComp:
            file_ = ""
            url = "https://example.com/song.mp3"

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == "https://example.com/song.mp3"

    def test_both_local_and_url(self):
        class MockComp:
            file_ = r"/tmp/song.mp3"
            url = "https://example.com/backup.mp3"

        local, url = resolve_component_ref(MockComp())
        assert local == "/tmp/song.mp3"
        assert url == "https://example.com/backup.mp3"

    def test_empty(self):
        class MockComp:
            file_ = ""
            url = ""

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == ""

    def test_none_values(self):
        class MockComp:
            file_ = None
            url = None

        local, url = resolve_component_ref(MockComp())
        assert local == ""
        assert url == ""


class TestReadAndValidate:
    async def test_valid_file(self, sample_audio_path):
        result = await _read_and_validate(
            str(sample_audio_path), "wav", 20, "test.wav"
        )
        assert result.filename == "test.wav"
        assert result.mime_type == "audio/wav"
        assert result.b64

    async def test_unsupported_extension_uses_generic_mime(self, temp_dir):
        p = temp_dir / "test.xyz"
        p.write_bytes(b"\x00" * 100)
        result = await _read_and_validate(str(p), "xyz", 20, "test.xyz")
        assert result.mime_type == "audio/xyz"

    async def test_file_too_large(self, sample_large_audio_path):
        with pytest.raises(AudioError, match="超过限制"):
            await _read_and_validate(
                str(sample_large_audio_path), "wav", 1, "large.wav"
            )

    async def test_file_not_exists(self):
        with pytest.raises(AudioError, match="不存在"):
            await _read_and_validate("/nonexistent/file.wav", "wav", 20, "file.wav")


class TestLoadAudioFromPath:
    async def test_valid(self, sample_audio_path):
        result = await load_audio_from_path(str(sample_audio_path), 20)
        assert result.b64

    async def test_file_not_found(self):
        with pytest.raises(AudioError, match="不存在"):
            await load_audio_from_path("/nonexistent.wav", 20)
