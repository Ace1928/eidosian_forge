from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

from ..forge_loader import ensure_forge_import

ensure_forge_import("knowledge_forge")
ensure_forge_import("memory_forge")

from knowledge_forge.core.graph import KnowledgeForge
from knowledge_forge.integrations.graphrag import GraphRAGIntegration

from ..core import tool
from ..transactions import (
    begin_transaction,
    find_latest_transaction_for_path,
    load_transaction,
)
from ..embeddings import SimpleEmbedder


FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", "/home/lloyd/eidosian_forge"))

try:
    from memory_forge import MemoryConfig, MemoryForge
except Exception:  # pragma: no cover - fallback for missing deps
    MemoryConfig = None
    MemoryForge = None

_embedder = SimpleEmbedder()
memory = None
if MemoryForge and MemoryConfig:
    memory = MemoryForge(
        config=MemoryConfig(
            episodic={
                "connection_string": str(FORGE_DIR / "data" / "semantic_memory.json"),
                "type": "json",
            }
        ),
        embedder=_embedder,
    )
kb = KnowledgeForge(persistence_path=FORGE_DIR / "data" / "kb.json")
grag = GraphRAGIntegration(graphrag_root=FORGE_DIR / "graphrag")
SEMANTIC_MEMORY_PATH = FORGE_DIR / "data" / "semantic_memory.json"


@tool(
    name="memory_add_semantic",
    description="Add a semantic memory entry to the knowledge memory store.",
    parameters={
        "type": "object",
        "properties": {"content": {"type": "string"}},
        "required": ["content"],
    },
)
def memory_add(content: str) -> str:
    """Add to episodic memory."""
    try:
        if memory is None:
            return "Error storing memory: memory backend unavailable"
        mid = memory.remember(content)
        return f"Stored memory: {mid}"
    except Exception as exc:
        return f"Error storing memory: {exc}"


@tool(
    description="Semantic search over memory.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def memory_search(query: str) -> str:
    """Semantic search over memory."""
    try:
        if memory is None:
            return "Error searching memory: memory backend unavailable"
        results = memory.recall(query)
        return "\n".join([f"- {r.content}" for r in results])
    except Exception as exc:
        return f"Error searching memory: {exc}"


@tool(
    name="kb_add",
    description="Add fact to the knowledge graph.",
    parameters={
        "type": "object",
        "properties": {
            "fact": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["fact", "tags"],
    },
)
def kb_add_fact(fact: str, tags: List[str]) -> str:
    """Add fact to Knowledge Graph."""
    persistence_path = getattr(kb, "persistence_path", None)
    if not isinstance(persistence_path, Path):
        persistence_path = None
    txn = begin_transaction("kb_add", [persistence_path]) if persistence_path else None
    try:
        node = kb.add_knowledge(fact, tags=tags)
        if txn:
            txn.commit()
            return f"Added node: {node.id} ({txn.id})"
        return f"Added node: {node.id}"
    except Exception as exc:
        if txn:
            txn.rollback(f"exception: {exc}")
        return f"Error adding knowledge: {exc}"


@tool(
    name="kb_search",
    description="Search the knowledge graph for matching content.",
)
def kb_search(query: str) -> str:
    """Search the knowledge graph for matching content."""
    results = kb.search(query)
    payload = [n.to_dict() for n in results]
    return str(payload)


@tool(
    name="kb_get_by_tag",
    description="Find knowledge nodes by tag.",
)
def kb_get_by_tag(tag: str) -> str:
    """Find knowledge nodes by tag."""
    results = kb.get_by_tag(tag)
    payload = [n.to_dict() for n in results]
    return str(payload)


@tool(
    name="kb_link",
    description="Create a bidirectional link between two knowledge nodes.",
)
def kb_link(node_id_a: str, node_id_b: str) -> str:
    """Create a bidirectional link between two knowledge nodes."""
    persistence_path = getattr(kb, "persistence_path", None)
    if not isinstance(persistence_path, Path):
        persistence_path = None
    txn = begin_transaction("kb_link", [persistence_path]) if persistence_path else None
    try:
        kb.link_nodes(node_id_a, node_id_b)
        if txn:
            txn.commit()
            return f"Linked {node_id_a} <-> {node_id_b} ({txn.id})"
        return f"Linked {node_id_a} <-> {node_id_b}"
    except Exception as exc:
        if txn:
            txn.rollback(f"exception: {exc}")
        return f"Error linking nodes: {exc}"


@tool(
    name="kb_delete",
    description="Delete a knowledge node by id.",
    parameters={
        "type": "object",
        "properties": {"node_id": {"type": "string"}},
        "required": ["node_id"],
    },
)
def kb_delete(node_id: str) -> str:
    """Delete a knowledge node by id."""
    persistence_path = getattr(kb, "persistence_path", None)
    if not isinstance(persistence_path, Path):
        persistence_path = None
    txn = begin_transaction("kb_delete", [persistence_path]) if persistence_path else None
    try:
        deleted = kb.delete_node(node_id)
        if not deleted:
            if txn:
                txn.rollback("no-op: not found")
            return "No-op: Not found"
        if txn:
            txn.commit()
            return f"Deleted node {node_id} ({txn.id})"
        return f"Deleted node {node_id}"
    except Exception as exc:
        if txn:
            txn.rollback(f"exception: {exc}")
        return f"Error deleting node: {exc}"


@tool(
    name="kb_restore",
    description="Restore knowledge base from a transaction snapshot.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
    },
)
def kb_restore(transaction_id: Optional[str] = None) -> str:
    """Restore knowledge base from a snapshot transaction."""
    persistence_path = getattr(kb, "persistence_path", None)
    if not isinstance(persistence_path, Path):
        persistence_path = None
    if not persistence_path:
        return "Error: Knowledge base persistence unavailable"
    txn_id = transaction_id or find_latest_transaction_for_path(persistence_path)
    if not txn_id:
        return "Error: No transaction found for knowledge base"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("kb_restore")
    try:
        kb.load()
    except Exception:
        pass
    return f"Knowledge base restored ({txn_id})"


@tool(
    name="memory_delete_semantic",
    description="Delete a semantic memory entry by id.",
)
def memory_delete_semantic(item_id: str) -> str:
    """Delete a semantic memory entry by id."""
    if memory is None:
        return "Error deleting memory: memory backend unavailable"
    txn = begin_transaction("memory_delete_semantic", [SEMANTIC_MEMORY_PATH])
    try:
        deleted = memory.episodic.delete(item_id)
        if not deleted:
            txn.rollback("no-op: not found")
            return "No-op: Not found"
        txn.commit()
        return f"Deleted ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error deleting memory: {exc}"


@tool(
    name="memory_clear_semantic",
    description="Clear the semantic memory store.",
)
def memory_clear_semantic() -> str:
    """Clear the semantic memory store."""
    if memory is None:
        return "Error clearing memory: memory backend unavailable"
    if memory.episodic.count() == 0:
        return "No-op: Memory already empty"
    txn = begin_transaction("memory_clear_semantic", [SEMANTIC_MEMORY_PATH])
    try:
        memory.episodic.clear()
        if memory.episodic.count() != 0:
            txn.rollback("verification_failed: not_empty")
            return f"Error: Verification failed; rolled back ({txn.id})"
        txn.commit()
        return f"Memory cleared ({txn.id})"
    except Exception as exc:
        txn.rollback(f"exception: {exc}")
        return f"Error clearing memory: {exc}"


@tool(
    name="memory_snapshot_semantic",
    description="Create a snapshot of the semantic memory store.",
    parameters={"type": "object", "properties": {}},
)
def memory_snapshot_semantic() -> str:
    """Create a snapshot of the semantic memory store."""
    txn = begin_transaction("memory_snapshot_semantic", [SEMANTIC_MEMORY_PATH])
    txn.commit("snapshot")
    return f"Snapshot created ({txn.id})"


@tool(
    name="memory_restore_semantic",
    description="Restore semantic memory from a snapshot transaction.",
    parameters={
        "type": "object",
        "properties": {"transaction_id": {"type": "string"}},
    },
)
def memory_restore_semantic(transaction_id: Optional[str] = None) -> str:
    """Restore semantic memory from a snapshot transaction."""
    txn_id = transaction_id or find_latest_transaction_for_path(SEMANTIC_MEMORY_PATH)
    if not txn_id:
        return "Error: No transaction found for semantic memory"
    txn = load_transaction(txn_id)
    if not txn:
        return "Error: Transaction not found"
    txn.rollback("memory_restore_semantic")
    try:
        if memory and hasattr(memory.episodic, "_load"):
            memory.episodic._load()
    except Exception:
        pass
    return f"Semantic memory restored ({txn_id})"


@tool(
    name="grag_query",
    description="Query GraphRAG.",
    parameters={
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
)
def rag_query(query: str) -> str:
    """Query GraphRAG."""
    return str(grag.global_query(query))


@tool(
    name="grag_index",
    description="Run a GraphRAG incremental index over scan roots.",
)
def grag_index(scan_roots: Optional[List[str]] = None) -> str:
    """Run a GraphRAG incremental index over scan roots."""
    roots = scan_roots or [str(FORGE_DIR)]
    root_paths = [Path(p) for p in roots]
    return str(grag.run_incremental_index(root_paths))


@tool(
    name="grag_query_local",
    description="Run a local GraphRAG query.",
)
def grag_query_local(query: str) -> str:
    """Run a local GraphRAG query."""
    return str(grag.local_query(query))
