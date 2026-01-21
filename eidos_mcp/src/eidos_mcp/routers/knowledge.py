from ..core import mcp
from memory_forge import MemoryForge, MemoryConfig
from llm_forge import OllamaProvider # Use Real Local Provider
from knowledge_forge import KnowledgeForge, GraphRAGIntegration
from pathlib import Path
import os

# Config
FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))

# Initialize Services
# Use nomic-embed-text for embeddings
embedder = OllamaProvider(embedding_model="nomic-embed-text")

# Use qwen2.5:0.5b for any generation tasks (future)
# generator = OllamaProvider(model="qwen2.5:0.5b")

memory = MemoryForge(
    config=MemoryConfig(
        episodic={"connection_string": str(FORGE_DIR / "data" / "memory_db"), "type": "chroma"}
    ),
    embedder=embedder
)
kb = KnowledgeForge(persistence_path=FORGE_DIR / "data" / "kb.json")
grag = GraphRAGIntegration(graphrag_root=FORGE_DIR / "graphrag")

@mcp.tool()
def memory_add(content: str) -> str:
    """Add to episodic memory."""
    try:
        mid = memory.remember(content)
        return f"Stored memory: {mid}"
    except Exception as e:
        return f"Error storing memory: {e}"

@mcp.tool()
def memory_search(query: str) -> str:
    """Semantic search over memory."""
    try:
        results = memory.recall(query)
        return "\n".join([f"- {r.content} ({r.importance})" for r in results])
    except Exception as e:
        return f"Error searching memory: {e}"

@mcp.tool()
def kb_add_fact(fact: str, tags: list[str]) -> str:
    """Add fact to Knowledge Graph."""
    node = kb.add_knowledge(fact, tags=tags)
    return f"Added node: {node.id}"

@mcp.tool()
def rag_query(query: str) -> str:
    """Query GraphRAG."""
    return str(grag.global_query(query))