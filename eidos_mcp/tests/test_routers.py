from eidos_mcp.routers import knowledge, memory


def test_memory_tools():
    memory_id = None
    semantic_id = None
    semantic_content = "Semantic MCP test entry"
    try:
        res = memory.memory_add("Eidos MCP test memory")
        assert "Memory added with ID" in res
        memory_id = res.split("Memory added with ID: ", 1)[1].split(" ", 1)[0]
        retrieved = memory.memory_retrieve("Eidos MCP test memory", limit=5)
        assert "Eidos MCP test memory" in retrieved
        add_semantic = knowledge.memory_add_semantic(semantic_content)
        assert "Stored semantic memory:" in add_semantic
        semantic_id = add_semantic.split("Stored semantic memory: ", 1)[1].split(" ", 1)[0]
        semantic_search = knowledge.memory_search("Semantic MCP test entry")
        assert semantic_content in semantic_search
    finally:
        if memory_id:
            memory.memory_delete(memory_id)
        if semantic_id:
            knowledge.memory_delete_semantic(semantic_id)


def test_kb_tools():
    node_id = None
    try:
        res = knowledge.kb_add_fact("Sky is blue", ["nature"])
        assert "Added node:" in res
        node_id = res.split("Added node: ", 1)[1].split(" ", 1)[0]
        search = knowledge.kb_search("Sky is blue")
        assert node_id in search or "Sky is blue" in search
    finally:
        if node_id:
            knowledge.kb_delete(node_id)
