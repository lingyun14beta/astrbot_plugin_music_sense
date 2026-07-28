"""Tests for llm_tools.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gemini_client import GeminiClient, GeminiClientError
from llm_tools import resolve_audio_ref, run_audio_analysis_from_path


@pytest.fixture
def mock_client():
    client = MagicMock(spec=GeminiClient)
    client.analyze = AsyncMock(return_value="这是一段优美的音乐。")
    client.close = AsyncMock()
    return client


class TestRunAudioAnalysisFromPath:
    async def test_success(self, sample_audio_path, mock_client):
        result = await run_audio_analysis_from_path(
            str(sample_audio_path), 20, mock_client
        )
        assert "优美的音乐" in result
        mock_client.analyze.assert_called_once()

    async def test_file_not_found(self, mock_client):
        result = await run_audio_analysis_from_path(
            "/nonexistent.wav", 20, mock_client
        )
        assert "音频分析失败" in result
        mock_client.analyze.assert_not_called()

    async def test_api_error(self, sample_audio_path, mock_client):
        mock_client.analyze.side_effect = GeminiClientError("API 超时")
        result = await run_audio_analysis_from_path(
            str(sample_audio_path), 20, mock_client
        )
        assert "API 超时" in result


class TestResolveAudioRef:
    async def test_local_file_exists(self, sample_audio_path):
        item = {
            "name": "test.wav",
            "ref": str(sample_audio_path),
            "is_local": True,
            "result": None,
        }
        result = await resolve_audio_ref(item, 20)
        assert result == str(sample_audio_path)

    async def test_local_file_gone(self):
        item = {
            "name": "gone.wav",
            "ref": "/nonexistent/gone.wav",
            "is_local": True,
            "result": None,
        }
        from audio_utils import AudioError
        with pytest.raises(AudioError, match="过期"):
            await resolve_audio_ref(item, 20)

    async def test_remote_downloads(self, sample_mp3_path):
        import asyncio
        item = {
            "name": "song.mp3",
            "ref": "https://example.com/song.mp3",
            "is_local": False,
            "result": None,
        }

        class MockResp:
            status = 200
            async def read(self):
                return sample_mp3_path.read_bytes()
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass

        class MockSession:
            async def __aenter__(self):
                return self
            async def __aexit__(self, *args):
                pass
            def get(self, url):
                return MockResp()

        with patch("audio_utils.aiohttp.ClientSession", return_value=MockSession()):
            result = await resolve_audio_ref(item, 20)
            assert Path(result).is_file()
            assert item["is_local"] is True
            assert item["ref"] == result

    async def test_empty_ref(self):
        item = {
            "name": "empty.wav",
            "ref": "",
            "is_local": False,
            "result": None,
        }
        from audio_utils import AudioError
        with pytest.raises(AudioError, match="为空"):
            await resolve_audio_ref(item, 20)
