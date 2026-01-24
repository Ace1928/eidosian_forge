import pytest
from unittest.mock import MagicMock, patch
from eidos_mcp.routers import knowledge, memory

@patch("eidos_mcp.routers.memory.memory")
def test_memory_tools(mock_memory):
    mock_memory.remember.return_value = "mem-123"
    mock_memory.recall.return_value = []
    
        res = memory.memory_add("Test memory")
        assert "mem-123" in res
    
        res = knowledge.memory_search("Test")
        assert "Semantic MCP test" in res  # Allow other memories to coexist
@patch("eidos_mcp.routers.knowledge.kb")
def test_kb_tools(mock_kb):
    mock_node = MagicMock()
    mock_node.id = "node-abc"
    mock_kb.add_knowledge.return_value = mock_node
    
    res = knowledge.kb_add_fact("Sky is blue", ["nature"])
    assert "node-abc" in res
