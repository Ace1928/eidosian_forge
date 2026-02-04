"""Tests for share link encoding/decoding."""
from __future__ import annotations

from pathlib import Path

from glyph_forge.cli.share_utils import encode_share_link, decode_share_link, load_share_link


def test_share_link_roundtrip_text() -> None:
    payload = "Glyph Forge".encode("utf-8")
    link = encode_share_link(payload, "txt", "demo.txt", source="unit-test")
    meta, data = decode_share_link(link)
    assert data == payload
    assert meta["format"] == "txt"


def test_share_link_roundtrip_binary_compressed() -> None:
    payload = b"x" * 4096
    link = encode_share_link(payload, "bin", "blob.bin", source="unit-test")
    meta, data = decode_share_link(link)
    assert data == payload
    assert meta["compression"] in (None, "gzip")


def test_load_share_link_from_file(tmp_path: Path) -> None:
    payload = b"hello"
    link = encode_share_link(payload, "txt", "demo.txt")
    link_path = tmp_path / "demo.gflink"
    link_path.write_text(link, encoding="utf-8")

    meta, data = load_share_link(str(link_path))
    assert data == payload
    assert meta["filename"] == "demo.txt"
