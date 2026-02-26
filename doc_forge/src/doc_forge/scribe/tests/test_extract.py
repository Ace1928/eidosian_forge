
import pytest
from doc_forge.scribe.extract import DocumentExtractor, is_text_bytes


def test_is_text_bytes():
    assert is_text_bytes(b"Hello world")
    assert is_text_bytes(b"")
    assert not is_text_bytes(b"Hello\x00world")


def test_extract_text(mock_config):
    extractor = DocumentExtractor(mock_config)
    p = mock_config.forge_root / "test.py"
    p.write_text("print('hello')", encoding="utf-8")

    content, meta = extractor.extract(p)
    assert content == "print('hello')"
    assert meta["doc_type"] == "py"


def test_extract_binary_error(mock_config):
    extractor = DocumentExtractor(mock_config)
    p = mock_config.forge_root / "test.bin"
    p.write_bytes(b"\x00\x01\x02")

    with pytest.raises(ValueError, match="binary-like"):
        extractor.extract(p)


def test_extract_large_file_truncation(mock_config):
    # Set small limit for testing
    mock_config.max_file_bytes = 10
    extractor = DocumentExtractor(mock_config)
    p = mock_config.forge_root / "large.txt"
    p.write_text("123456789012345", encoding="utf-8")

    # It reads bytes first, so it should read 10 bytes then decode
    content, _ = extractor.extract(p)
    assert len(content.encode("utf-8")) <= 10
