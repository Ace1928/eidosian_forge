from __future__ import annotations

from pathlib import Path
from urllib.error import URLError

import crawl_forge.tika_extractor as tika_mod
from crawl_forge.tika_extractor import TikaExtractor


class _ParserNoFromURL:
    def from_buffer(self, data: bytes) -> dict:
        assert isinstance(data, bytes)
        return {"content": "buffer-content", "metadata": {"source": "buffer"}}


class _ParserWithFromURL:
    def from_url(self, url: str) -> dict:
        return {"content": f"url-content:{url}", "metadata": {"source": "url"}}

    def from_buffer(self, data: bytes) -> dict:
        return {"content": "buffer-fallback", "metadata": {}}


class _ParserFromURLRaises:
    def from_url(self, url: str) -> dict:
        raise RuntimeError("from_url boom")

    def from_buffer(self, data: bytes) -> dict:
        return {"content": "buffer-after-fail", "metadata": {"fallback": "yes"}}


def test_extract_from_url_falls_back_to_buffer_when_from_url_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(tika_mod, "_get_tika_parser", lambda: _ParserNoFromURL())
    extractor = TikaExtractor(cache_dir=tmp_path, enable_cache=False)
    monkeypatch.setattr(
        extractor,
        "_download_url_bytes",
        lambda url, timeout=20: (b"hello", {"content_type": "text/plain", "content_length": "5"}),
    )

    out = extractor.extract_from_url("https://example.com", use_cache=False)
    assert out["status"] == "success"
    assert out["content"] == "buffer-content"
    assert out["metadata"]["source_url"] == "https://example.com"
    assert out["metadata"]["fetch_content_type"] == "text/plain"


def test_extract_from_url_uses_from_url_when_available(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tika_mod, "_get_tika_parser", lambda: _ParserWithFromURL())
    extractor = TikaExtractor(cache_dir=tmp_path, enable_cache=False)

    out = extractor.extract_from_url("https://example.com", use_cache=False)
    assert out["status"] == "success"
    assert out["content"] == "url-content:https://example.com"
    assert out["metadata"]["source"] == "url"


def test_extract_from_url_falls_back_when_from_url_raises(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(tika_mod, "_get_tika_parser", lambda: _ParserFromURLRaises())
    extractor = TikaExtractor(cache_dir=tmp_path, enable_cache=False)
    monkeypatch.setattr(
        extractor,
        "_download_url_bytes",
        lambda url, timeout=20: (b"fallback", {"content_type": "application/octet-stream", "content_length": "8"}),
    )

    out = extractor.extract_from_url("https://example.com", use_cache=False)
    assert out["status"] == "success"
    assert out["content"] == "buffer-after-fail"
    assert out["metadata"]["fallback"] == "yes"
    assert out["metadata"]["source_url"] == "https://example.com"


def test_download_url_bytes_retries_with_unverified_context_on_ssl_error(
    tmp_path: Path, monkeypatch
) -> None:
    class _Resp:
        def __init__(self) -> None:
            self.headers = {"Content-Type": "text/plain", "Content-Length": "2"}

        def read(self) -> bytes:
            return b"ok"

        def __enter__(self) -> "_Resp":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    monkeypatch.setattr(tika_mod, "_get_tika_parser", lambda: _ParserNoFromURL())
    extractor = TikaExtractor(cache_dir=tmp_path, enable_cache=False)
    monkeypatch.setattr(extractor, "_allow_insecure_ssl_fallback", lambda: True)

    calls = {"count": 0}

    def _fake_urlopen(req, timeout=20, context=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise URLError("[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed")
        assert context is not None
        return _Resp()

    monkeypatch.setattr(tika_mod, "urlopen", _fake_urlopen)

    body, meta = extractor._download_url_bytes("https://example.com")
    assert body == b"ok"
    assert meta["ssl_verify"] == "false"
