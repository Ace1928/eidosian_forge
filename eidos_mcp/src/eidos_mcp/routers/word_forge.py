"""
Word Forge Router for EIDOS MCP Server.

Exposes semantic graph capabilities using NetworkX directly.
The full word_forge has circular import issues - this provides
a functional subset for semantic graph operations.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import networkx as nx
from eidosian_core import eidosian

from .. import FORGE_ROOT
from ..core import tool

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))).resolve()
SEMANTIC_GRAPH_PATH = FORGE_DIR / "data" / "eidos_semantic_graph.json"
COORDINATOR_STATUS_PATH = FORGE_DIR / "data" / "runtime" / "forge_coordinator_status.json"

# Simple in-memory graph
_graph: Optional[nx.Graph] = None
TERM_ENRICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "definition": {"type": "string"},
        "pos": {"type": "string"},
        "aliases": {"type": "array", "items": {"type": "string"}},
        "domains": {"type": "array", "items": {"type": "string"}},
        "related_terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "relation_type": {"type": "string"},
                    "weight": {"type": "number"},
                },
                "required": ["term", "relation_type", "weight"],
            },
        },
    },
    "required": ["definition", "pos", "aliases", "domains", "related_terms"],
}
TEXT_LEXICON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "terms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "pos": {"type": "string"},
                    "domains": {"type": "array", "items": {"type": "string"}},
                    "related_terms": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "term": {"type": "string"},
                                "relation_type": {"type": "string"},
                                "weight": {"type": "number"},
                            },
                            "required": ["term", "relation_type", "weight"],
                        },
                    },
                },
                "required": ["term", "definition", "pos", "domains", "related_terms"],
            },
        }
    },
    "required": ["terms"],
}


def _get_graph() -> nx.Graph:
    """Get or create the semantic graph."""
    global _graph
    if _graph is None:
        _graph = nx.Graph()
        # Load from file if exists
        if SEMANTIC_GRAPH_PATH.exists():
            try:
                data = json.loads(SEMANTIC_GRAPH_PATH.read_text())
                for node in data.get("nodes", []):
                    node_id = node.pop("id")
                    _graph.add_node(node_id, **node)
                for edge in data.get("edges", []):
                    source = edge.pop("source")
                    target = edge.pop("target")
                    _graph.add_edge(source, target, **edge)
            except Exception:
                pass
    return _graph


def _save_graph() -> None:
    """Persist graph to file."""
    graph = _get_graph()
    SEMANTIC_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "nodes": [{**graph.nodes[n], "id": n} for n in graph.nodes()],
        "edges": [{"source": u, "target": v, **graph.edges[u, v]} for u, v in graph.edges()],
    }
    SEMANTIC_GRAPH_PATH.write_text(json.dumps(data, indent=2))


def _extract_json_fragment(text: str) -> Dict[str, Any] | None:
    candidate = str(text or "").strip()
    if not candidate:
        return None
    try:
        payload = json.loads(candidate)
        return payload if isinstance(payload, dict) else None
    except Exception:
        pass
    match = re.search(r"\{.*\}", candidate, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _get_model_config():
    try:
        from eidos_mcp.config.models import get_model_config

        return get_model_config()
    except Exception:
        return None


def _coordinator_budget(model: str, role: str) -> dict[str, Any]:
    try:
        from eidosian_runtime import ForgeRuntimeCoordinator

        coordinator = ForgeRuntimeCoordinator(COORDINATOR_STATUS_PATH)
        decision = coordinator.can_allocate(
            owner="word_forge_router",
            requested_models=[{"family": "ollama", "model": str(model), "role": str(role)}],
            allow_same_owner=False,
        )
        return {
            "allowed": bool(decision.get("allowed")),
            "reason": str(decision.get("reason") or ""),
            "decision": decision,
        }
    except Exception:
        return {"allowed": True, "reason": "coordinator_unavailable", "decision": {}}


def _generate_structured_payload(
    *,
    prompt: str,
    system: str,
    schema: Dict[str, Any],
    thinking_mode: str = "on",
    timeout_sec: float = 900.0,
    max_tokens: int = 700,
) -> Dict[str, Any]:
    cfg = _get_model_config()
    if cfg is None or not getattr(cfg, "is_ollama_running", lambda: False)():
        return {}
    model_name = str(getattr(cfg, "inference_model", "") or "qwen3.5:2b")
    budget = _coordinator_budget(model_name, role="word_forge_enrichment")
    if not budget.get("allowed"):
        denied = {
            "_budget_denied": True,
            "_budget_reason": budget.get("reason"),
            "_effective_thinking_mode": "off",
        }
        return denied
    requested_mode = str(thinking_mode or "on")
    for mode in [requested_mode, "off"] if requested_mode != "off" else ["off"]:
        try:
            payload = cfg.generate_payload(
                prompt,
                model=model_name,
                system=system,
                thinking_mode=mode,
                timeout=timeout_sec,
                max_tokens=max_tokens,
                temperature=0.1,
                format=schema,
            )
        except Exception:
            continue
        parsed = _extract_json_fragment(str(payload.get("response") or ""))
        if parsed:
            parsed["_effective_thinking_mode"] = mode
            return parsed
    return {}


# =============================================================================
# GRAPH BUILDING TOOLS
# =============================================================================


@tool(
    name="wf_add_term",
    description="Add a term (word/phrase) to the Word Forge semantic graph.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "The term to add"},
            "definition": {"type": "string", "description": "Definition or meaning of the term"},
            "pos": {"type": "string", "description": "Part of speech (noun, verb, adjective, etc.)"},
            "source": {"type": "string", "description": "Source of the definition"},
        },
        "required": ["term"],
    },
)
@eidosian()
def wf_add_term(
    term: str,
    definition: Optional[str] = None,
    pos: Optional[str] = None,
    source: Optional[str] = None,
) -> str:
    """Add a term to the semantic graph."""
    graph = _get_graph()

    attributes = {}
    if definition:
        attributes["definition"] = definition
    if pos:
        attributes["pos"] = pos
    if source:
        attributes["source"] = source

    graph.add_node(term, **attributes)
    _save_graph()
    return f"Added term: {term}"


@tool(
    name="wf_add_relationship",
    description="Add a semantic relationship between two terms.",
    parameters={
        "type": "object",
        "properties": {
            "term1": {"type": "string", "description": "First term"},
            "term2": {"type": "string", "description": "Second term"},
            "relation_type": {
                "type": "string",
                "description": "Type of relationship (synonym, antonym, hypernym, hyponym, related, association)",
            },
            "weight": {"type": "number", "description": "Strength of relationship (0.0-1.0)"},
        },
        "required": ["term1", "term2", "relation_type"],
    },
)
@eidosian()
def wf_add_relationship(
    term1: str,
    term2: str,
    relation_type: str,
    weight: float = 0.5,
) -> str:
    """Add a relationship between terms."""
    graph = _get_graph()

    # Ensure both terms exist
    if not graph.has_node(term1):
        graph.add_node(term1)
    if not graph.has_node(term2):
        graph.add_node(term2)

    graph.add_edge(term1, term2, type=relation_type, weight=weight)
    _save_graph()
    return f"Added {relation_type} relationship: {term1} <-> {term2} (weight={weight})"


# =============================================================================
# QUERY TOOLS
# =============================================================================


@tool(
    name="wf_get_term",
    description="Get information about a term from the semantic graph.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "The term to look up"},
        },
        "required": ["term"],
    },
)
@eidosian()
def wf_get_term(term: str) -> str:
    """Get term information."""
    graph = _get_graph()

    if not graph.has_node(term):
        return f"Term not found: {term}"

    node_data = dict(graph.nodes[term])
    neighbors = []
    for neighbor in graph.neighbors(term):
        edge_data = graph.edges[term, neighbor]
        neighbors.append(
            {
                "term": neighbor,
                "type": edge_data.get("type", "related"),
                "weight": edge_data.get("weight", 0.5),
            }
        )

    result = {
        "term": term,
        "attributes": node_data,
        "relationships": neighbors,
    }
    return json.dumps(result, indent=2)


@tool(
    name="wf_find_related",
    description="Find terms semantically related to a given term.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "The term to find relations for"},
            "relation_type": {"type": "string", "description": "Filter by relationship type (optional)"},
            "limit": {"type": "integer", "description": "Maximum results (default: 10)"},
        },
        "required": ["term"],
    },
)
@eidosian()
def wf_find_related(
    term: str,
    relation_type: Optional[str] = None,
    limit: int = 10,
) -> str:
    """Find related terms."""
    graph = _get_graph()

    if not graph.has_node(term):
        return f"Term not found: {term}"

    results = []
    for neighbor in graph.neighbors(term):
        edge_data = graph.edges[term, neighbor]
        edge_type = edge_data.get("type", "related")
        if relation_type and edge_type != relation_type:
            continue
        results.append(
            {
                "term": neighbor,
                "type": edge_type,
                "weight": edge_data.get("weight", 0.5),
            }
        )

    # Sort by weight
    results.sort(key=lambda x: x["weight"], reverse=True)
    results = results[:limit]

    if not results:
        return f"No related terms found for: {term}"

    return json.dumps(
        {
            "term": term,
            "related": results,
            "count": len(results),
        },
        indent=2,
    )


@tool(
    name="wf_search_terms",
    description="Search for terms matching a pattern.",
    parameters={
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Search pattern (supports * wildcards)"},
            "limit": {"type": "integer", "description": "Maximum results (default: 20)"},
        },
        "required": ["pattern"],
    },
)
@eidosian()
def wf_search_terms(pattern: str, limit: int = 20) -> str:
    """Search for terms by pattern."""
    import fnmatch

    graph = _get_graph()
    all_nodes = list(graph.nodes())
    matches = [n for n in all_nodes if fnmatch.fnmatch(n.lower(), pattern.lower())][:limit]

    return json.dumps(
        {
            "pattern": pattern,
            "matches": matches,
            "count": len(matches),
        },
        indent=2,
    )


# =============================================================================
# ANALYSIS TOOLS
# =============================================================================


@tool(
    name="wf_graph_stats",
    description="Get statistics about the Word Forge semantic graph.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def wf_graph_stats() -> str:
    """Get graph statistics."""
    graph = _get_graph()

    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4) if graph.number_of_nodes() > 1 else 0,
        "connected_components": nx.number_connected_components(graph) if graph.number_of_nodes() > 0 else 0,
    }

    # Count edge types
    edge_types: Dict[str, int] = {}
    for _, _, data in graph.edges(data=True):
        edge_type = data.get("type", "unknown")
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1
    stats["edge_types"] = edge_types

    return json.dumps(stats, indent=2)


@tool(
    name="wf_find_path",
    description="Find semantic path between two terms.",
    parameters={
        "type": "object",
        "properties": {
            "term1": {"type": "string", "description": "Starting term"},
            "term2": {"type": "string", "description": "Ending term"},
        },
        "required": ["term1", "term2"],
    },
)
@eidosian()
def wf_find_path(term1: str, term2: str) -> str:
    """Find path between terms."""
    graph = _get_graph()

    if not graph.has_node(term1):
        return f"Term not found: {term1}"
    if not graph.has_node(term2):
        return f"Term not found: {term2}"

    try:
        path = nx.shortest_path(graph, term1, term2)
        return json.dumps(
            {
                "from": term1,
                "to": term2,
                "path": path,
                "length": len(path) - 1,
            },
            indent=2,
        )
    except nx.NetworkXNoPath:
        return f"No path found between '{term1}' and '{term2}'"


# =============================================================================
# PERSISTENCE TOOLS
# =============================================================================


@tool(
    name="wf_save_graph",
    description="Save the semantic graph to disk.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def wf_save_graph() -> str:
    """Save the graph."""
    graph = _get_graph()
    _save_graph()
    stats = {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
    }
    return f"Graph saved ({stats['nodes']} nodes, {stats['edges']} edges)"


@tool(
    name="wf_enrich_term",
    description="Use the configured reasoning model to enrich a Word Forge term with definitions and relationships.",
    parameters={
        "type": "object",
        "properties": {
            "term": {"type": "string", "description": "The term to enrich"},
            "context": {"type": "string", "description": "Optional grounding context"},
            "thinking_mode": {"type": "string", "description": "Reasoning mode (default: on)"},
            "timeout_sec": {"type": "number", "description": "Timeout in seconds (default: 900)"},
        },
        "required": ["term"],
    },
)
@eidosian()
def wf_enrich_term(term: str, context: str = "", thinking_mode: str = "on", timeout_sec: float = 900.0) -> str:
    graph = _get_graph()
    existing = dict(graph.nodes[term]) if graph.has_node(term) else {}
    payload = _generate_structured_payload(
        prompt=(
            f"Term: {term}\n"
            f"Existing attributes: {json.dumps(existing, ensure_ascii=True)}\n"
            f"Context: {context[:2000]}"
        ),
        system=(
            "You are enriching a lexical graph entry for the Eidosian Forge. "
            "Use only grounded lexical knowledge and the provided context. Return strict JSON."
        ),
        schema=TERM_ENRICH_SCHEMA,
        thinking_mode=thinking_mode,
        timeout_sec=timeout_sec,
    )
    if not payload:
        return json.dumps({"status": "skipped", "reason": "model unavailable or returned no structured payload"}, indent=2)
    if payload.get("_budget_denied"):
        return json.dumps(
            {
                "status": "skipped",
                "reason": "runtime budget denied",
                "budget_reason": payload.get("_budget_reason"),
            },
            indent=2,
        )

    attributes = dict(existing)
    for key in ("definition", "pos"):
        if str(payload.get(key) or "").strip():
            attributes[key] = str(payload.get(key)).strip()
    for key in ("aliases", "domains"):
        value = payload.get(key)
        if isinstance(value, list):
            attributes[key] = [str(item).strip() for item in value if str(item).strip()]
    attributes["source"] = "llm_enrichment"
    attributes["effective_thinking_mode"] = payload.get("_effective_thinking_mode")

    graph.add_node(term, **attributes)
    created_edges = 0
    for row in payload.get("related_terms", []) or []:
        if not isinstance(row, dict):
            continue
        other = str(row.get("term") or "").strip()
        rel_type = str(row.get("relation_type") or "related").strip() or "related"
        if not other:
            continue
        if not graph.has_node(other):
            graph.add_node(other, source="llm_enrichment")
        graph.add_edge(term, other, type=rel_type, weight=float(row.get("weight") or 0.5))
        created_edges += 1

    _save_graph()
    return json.dumps(
        {
            "status": "success",
            "term": term,
            "definition": attributes.get("definition"),
            "pos": attributes.get("pos"),
            "aliases": attributes.get("aliases", []),
            "domains": attributes.get("domains", []),
            "relationships_added": created_edges,
            "effective_thinking_mode": payload.get("_effective_thinking_mode"),
        },
        indent=2,
    )


@tool(
    name="wf_build_from_text",
    description="Build semantic relationships from a text passage.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to analyze and build graph from"},
            "min_word_length": {"type": "integer", "description": "Minimum word length to include (default: 4)"},
        },
        "required": ["text"],
    },
)
@eidosian()
def wf_build_from_text(text: str, min_word_length: int = 4) -> str:
    """Build graph from text."""
    import re

    graph = _get_graph()

    # Extract unique words
    words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    unique_words = [w for w in set(words) if len(w) >= min_word_length]

    nodes_added = 0
    edges_added = 0

    # Add words as nodes
    for word in unique_words[:100]:  # Limit to 100 terms
        if not graph.has_node(word):
            graph.add_node(word, source="text_extraction")
            nodes_added += 1

    # Create co-occurrence relationships for words that appear close together
    word_positions = {}
    for i, word in enumerate(words):
        if word in unique_words:
            word_positions.setdefault(word, []).append(i)

    for word1 in list(unique_words)[:50]:
        for word2 in unique_words:
            if word1 >= word2:
                continue
            # Check if they co-occur within a window
            pos1 = set(word_positions.get(word1, []))
            pos2 = set(word_positions.get(word2, []))

            co_occurs = False
            for p1 in pos1:
                for p2 in pos2:
                    if abs(p1 - p2) <= 5:  # Within 5 words
                        co_occurs = True
                        break
                if co_occurs:
                    break

            if co_occurs and not graph.has_edge(word1, word2):
                graph.add_edge(word1, word2, type="co_occurrence", weight=0.3)
                edges_added += 1

    _save_graph()

    return json.dumps(
        {
            "status": "success",
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
        },
        indent=2,
    )


@tool(
    name="wf_build_lexicon_from_text",
    description="Use the configured reasoning model to extract enriched lexical terms and relationships from text.",
    parameters={
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Grounding text"},
            "thinking_mode": {"type": "string", "description": "Reasoning mode (default: on)"},
            "timeout_sec": {"type": "number", "description": "Timeout in seconds (default: 900)"},
        },
        "required": ["text"],
    },
)
@eidosian()
def wf_build_lexicon_from_text(text: str, thinking_mode: str = "on", timeout_sec: float = 900.0) -> str:
    graph = _get_graph()
    payload = _generate_structured_payload(
        prompt=text[:6000],
        system=(
            "Extract a compact lexical graph from the provided text for the Eidosian Forge. "
            "Return terms, short grounded definitions, domains, and weighted semantic relationships as strict JSON."
        ),
        schema=TEXT_LEXICON_SCHEMA,
        thinking_mode=thinking_mode,
        timeout_sec=timeout_sec,
        max_tokens=1200,
    )
    if not payload:
        return wf_build_from_text(text)
    if payload.get("_budget_denied"):
        fallback = json.loads(wf_build_from_text(text))
        fallback["budget_denied"] = True
        fallback["budget_reason"] = payload.get("_budget_reason")
        fallback["effective_thinking_mode"] = payload.get("_effective_thinking_mode")
        return json.dumps(fallback, indent=2)

    nodes_added = 0
    edges_added = 0
    for item in payload.get("terms", []) or []:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term") or "").strip().lower()
        if not term:
            continue
        attrs = {
            "definition": str(item.get("definition") or "").strip(),
            "pos": str(item.get("pos") or "").strip(),
            "domains": [str(x).strip() for x in (item.get("domains") or []) if str(x).strip()],
            "source": "llm_text_extraction",
            "effective_thinking_mode": payload.get("_effective_thinking_mode"),
        }
        if not graph.has_node(term):
            nodes_added += 1
        graph.add_node(term, **attrs)
        for rel in item.get("related_terms", []) or []:
            if not isinstance(rel, dict):
                continue
            other = str(rel.get("term") or "").strip().lower()
            rel_type = str(rel.get("relation_type") or "related").strip() or "related"
            if not other:
                continue
            if not graph.has_node(other):
                graph.add_node(other, source="llm_text_extraction")
                nodes_added += 1
            if not graph.has_edge(term, other):
                edges_added += 1
            graph.add_edge(term, other, type=rel_type, weight=float(rel.get("weight") or 0.5))

    _save_graph()
    return json.dumps(
        {
            "status": "success",
            "nodes_added": nodes_added,
            "edges_added": edges_added,
            "total_nodes": graph.number_of_nodes(),
            "total_edges": graph.number_of_edges(),
            "effective_thinking_mode": payload.get("_effective_thinking_mode"),
        },
        indent=2,
    )
