from unittest.mock import Mock

import pytest
from memory_forge.compression.compressor import MemoryCompressor
from memory_forge.core.interfaces import MemoryItem


def test_compressor():
    mock_llm = Mock()
    mock_llm.summarize.return_value = "Summary of events"

    compressor = MemoryCompressor(mock_llm)

    items = [MemoryItem("Event A"), MemoryItem("Event B")]

    result = compressor.compress_batch(items)

    assert result.content == "Summary of events"
    assert result.metadata["is_compressed"] is True
    assert len(result.metadata["source_ids"]) == 2
    mock_llm.summarize.assert_called_once()


def test_compress_empty():
    compressor = MemoryCompressor(Mock())
    with pytest.raises(ValueError):
        compressor.compress_batch([])
