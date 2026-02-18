"""
Tiered Memory Router for EIDOS MCP Server.

Provides tools for interacting with the multi-tiered memory system:
- SHORT_TERM: Session-scoped, 1-hour TTL
- WORKING: Task-relevant, 24-hour TTL
- LONG_TERM: Permanent episodic memories
- SELF: EIDOS identity, lessons, growth (permanent)
- USER: User preferences and patterns (permanent)
"""

from __future__ import annotations
from eidosian_core import eidosian

import json
import os
from pathlib import Path
from typing import Optional, List

from ..core import tool
from .. import FORGE_ROOT
from ..forge_loader import ensure_forge_import

ensure_forge_import("memory_forge")

try:
    from memory_forge import (
        TieredMemorySystem,
        TieredMemoryItem,
        MemoryTier,
        MemoryNamespace,
    )
    TIERED_AVAILABLE = True
except ImportError:
    TIERED_AVAILABLE = False
    TieredMemorySystem = None
    TieredMemoryItem = None
    MemoryTier = None
    MemoryNamespace = None

FORGE_DIR = Path(os.environ.get("EIDOS_FORGE_DIR", str(FORGE_ROOT))).resolve()
TIERED_MEMORY_DIR = Path(
    os.environ.get("EIDOS_TIERED_MEMORY_DIR", FORGE_DIR / "data" / "tiered_memory")
)

# Initialize the tiered memory system
_tiered_memory: Optional[TieredMemorySystem] = None


def _get_tiered_memory() -> Optional[TieredMemorySystem]:
    """Lazy-load the tiered memory system."""
    global _tiered_memory
    if _tiered_memory is None and TIERED_AVAILABLE:
        _tiered_memory = TieredMemorySystem(persistence_dir=TIERED_MEMORY_DIR)
    return _tiered_memory


# =============================================================================
# SELF-MEMORY TOOLS (EIDOS Identity, Lessons, Growth)
# =============================================================================


@tool(
    name="eidos_remember_self",
    description="Store a memory about EIDOS itself - identity, capabilities, lessons learned, growth.",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The self-knowledge content to remember"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorizing this self-memory (e.g., 'identity', 'lesson', 'capability')"
            },
        },
        "required": ["content"],
    },
)
@eidosian()
def eidos_remember_self(
    content: str,
    tags: Optional[List[str]] = None,
) -> str:
    """Store EIDOS self-memory."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tag_set = set(tags) if tags else set()
    mid = mem.remember_self(content, tags=tag_set)
    return f"Self-memory stored: {mid}"


@tool(
    name="eidos_recall_self",
    description="Recall EIDOS self-memories matching a query.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search self-memories"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results to return (default 5)"
            },
        },
        "required": ["query"],
    },
)
@eidosian()
def eidos_recall_self(query: str, limit: int = 5) -> str:
    """Recall EIDOS self-memories."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    results = mem.recall_self(query, limit=limit)
    if not results:
        return "No self-memories found matching query"
    
    output = []
    for item in results:
        output.append({
            "id": item.id,
            "content": item.content,
            "tags": list(item.tags),
            "importance": item.importance,
            "access_count": item.access_count,
        })
    return json.dumps(output, indent=2)


@tool(
    name="eidos_remember_lesson",
    description="Store a lesson learned by EIDOS from an experience or error.",
    parameters={
        "type": "object",
        "properties": {
            "lesson": {
                "type": "string",
                "description": "The lesson learned"
            },
            "context": {
                "type": "string",
                "description": "Context in which the lesson was learned"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Additional categorization tags"
            },
        },
        "required": ["lesson"],
    },
)
@eidosian()
def eidos_remember_lesson(
    lesson: str,
    context: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> str:
    """Store a lesson learned by EIDOS."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tag_set = set(tags) if tags else set()
    mid = mem.remember_lesson(lesson, context=context, tags=tag_set)
    return f"Lesson stored: {mid}"


# =============================================================================
# USER-MEMORY TOOLS
# =============================================================================


@tool(
    name="eidos_remember_user",
    description="Store a memory about a user - preferences, patterns, context.",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Information about the user to remember"
            },
            "user_id": {
                "type": "string",
                "description": "User identifier (default: 'lloyd')"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorizing this user memory"
            },
        },
        "required": ["content"],
    },
)
@eidosian()
def eidos_remember_user(
    content: str,
    user_id: str = "lloyd",
    tags: Optional[List[str]] = None,
) -> str:
    """Store user-related memory."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tag_set = set(tags) if tags else set()
    mid = mem.remember_user(content, user_id=user_id, tags=tag_set)
    return f"User memory stored: {mid}"


@tool(
    name="eidos_recall_user",
    description="Recall memories about a specific user.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query to search user memories"
            },
            "user_id": {
                "type": "string",
                "description": "User identifier (default: 'lloyd')"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 5)"
            },
        },
        "required": ["query"],
    },
)
@eidosian()
def eidos_recall_user(
    query: str,
    user_id: str = "lloyd",
    limit: int = 5,
) -> str:
    """Recall user-related memories."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    results = mem.recall_user(query, user_id=user_id, limit=limit)
    if not results:
        return f"No memories found for user '{user_id}' matching query"
    
    output = []
    for item in results:
        output.append({
            "id": item.id,
            "content": item.content,
            "tags": list(item.tags),
            "access_count": item.access_count,
        })
    return json.dumps(output, indent=2)


# =============================================================================
# GENERAL TIERED MEMORY TOOLS
# =============================================================================


@tool(
    name="tiered_remember",
    description="Store a memory with explicit tier and namespace control.",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content to remember"
            },
            "tier": {
                "type": "string",
                "enum": ["short_term", "working", "long_term", "self", "user"],
                "description": "Memory tier (default: working)"
            },
            "namespace": {
                "type": "string",
                "enum": ["eidos", "user", "task", "knowledge", "code", "conversation"],
                "description": "Namespace (default: task)"
            },
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Tags for categorization"
            },
            "importance": {
                "type": "number",
                "description": "Importance (0.0-1.0)"
            },
        },
        "required": ["content"],
    },
)
@eidosian()
def tiered_remember(
    content: str,
    tier: str = "working",
    namespace: str = "task",
    tags: Optional[List[str]] = None,
    importance: float = 0.5,
) -> str:
    """Store memory with explicit tier/namespace."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tier_map = {
        "short_term": MemoryTier.SHORT_TERM,
        "working": MemoryTier.WORKING,
        "long_term": MemoryTier.LONG_TERM,
        "self": MemoryTier.SELF,
        "user": MemoryTier.USER,
    }
    namespace_map = {
        "eidos": MemoryNamespace.EIDOS,
        "user": MemoryNamespace.USER,
        "task": MemoryNamespace.TASK,
        "knowledge": MemoryNamespace.KNOWLEDGE,
        "code": MemoryNamespace.CODE,
        "conversation": MemoryNamespace.CONVERSATION,
    }
    
    tag_set = set(tags) if tags else set()
    mid = mem.remember(
        content=content,
        tier=tier_map.get(tier, MemoryTier.WORKING),
        namespace=namespace_map.get(namespace, MemoryNamespace.TASK),
        tags=tag_set,
        importance=importance,
    )
    return f"Memory stored: {mid} (tier={tier}, namespace={namespace})"


@tool(
    name="tiered_recall",
    description="Recall memories across tiers and namespaces.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "tier": {
                "type": "string",
                "enum": ["short_term", "working", "long_term", "self", "user", "all"],
                "description": "Filter by tier (default: all)"
            },
            "namespace": {
                "type": "string",
                "enum": ["eidos", "user", "task", "knowledge", "code", "conversation", "all"],
                "description": "Filter by namespace (default: all)"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum results (default 10)"
            },
        },
        "required": ["query"],
    },
)
@eidosian()
def tiered_recall(
    query: str,
    tier: str = "all",
    namespace: str = "all",
    limit: int = 10,
) -> str:
    """Recall memories with optional tier/namespace filtering."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tier_map = {
        "short_term": MemoryTier.SHORT_TERM,
        "working": MemoryTier.WORKING,
        "long_term": MemoryTier.LONG_TERM,
        "self": MemoryTier.SELF,
        "user": MemoryTier.USER,
    }
    namespace_map = {
        "eidos": MemoryNamespace.EIDOS,
        "user": MemoryNamespace.USER,
        "task": MemoryNamespace.TASK,
        "knowledge": MemoryNamespace.KNOWLEDGE,
        "code": MemoryNamespace.CODE,
        "conversation": MemoryNamespace.CONVERSATION,
    }
    
    filter_tier = [tier_map.get(tier)] if tier != "all" and tier in tier_map else None
    filter_namespace = [namespace_map.get(namespace)] if namespace != "all" and namespace in namespace_map else None
    
    results = mem.recall(
        query=query,
        tiers=filter_tier,
        namespaces=filter_namespace,
        limit=limit,
    )
    
    if not results:
        return "No memories found matching query"
    
    output = []
    for item in results:
        output.append({
            "id": item.id,
            "content": item.content,
            "tier": item.tier.value,
            "namespace": item.namespace.value,
            "tags": list(item.tags),
            "importance": item.importance,
            "access_count": item.access_count,
        })
    return json.dumps(output, indent=2)


@tool(
    name="tiered_memory_stats",
    description="Get statistics about the tiered memory system.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def tiered_memory_stats() -> str:
    """Return tiered memory statistics."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    stats = mem.stats()
    return json.dumps(stats, indent=2)


@tool(
    name="tiered_memory_cleanup",
    description="Clean up expired memories and persist changes.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def tiered_memory_cleanup() -> str:
    """Clean up expired memories."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"

    before = mem.stats()["total"]
    if hasattr(mem, "cleanup_expired"):
        removed = int(mem.cleanup_expired())
    elif hasattr(mem, "cleanup"):
        # Backward compatibility for older TieredMemorySystem implementations.
        mem.cleanup()  # type: ignore[attr-defined]
        after = mem.stats()["total"]
        removed = before - after
    else:
        return "Error: Tiered memory cleanup not supported by current memory backend"

    if hasattr(mem, "save_all"):
        try:
            mem.save_all()  # type: ignore[attr-defined]
        except Exception:
            pass

    return f"Cleanup complete: removed {removed} expired memories"


@tool(
    name="tiered_promote_memory",
    description="Promote a memory to a higher tier (e.g., short_term -> working).",
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to promote"
            },
            "target_tier": {
                "type": "string",
                "enum": ["working", "long_term"],
                "description": "Target tier"
            },
        },
        "required": ["memory_id", "target_tier"],
    },
)
@eidosian()
def tiered_promote_memory(memory_id: str, target_tier: str) -> str:
    """Promote a memory to a higher tier."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    tier_map = {
        "working": MemoryTier.WORKING,
        "long_term": MemoryTier.LONG_TERM,
    }
    target = tier_map.get(target_tier)
    if not target:
        return f"Error: Invalid target tier '{target_tier}'"
    
    success = mem.promote(memory_id, target)
    if success:
        return f"Memory {memory_id} promoted to {target_tier}"
    return f"Error: Could not promote memory {memory_id}"


# =============================================================================
# CONTEXT SUGGESTION (Foundation for autonomous context)
# =============================================================================


@tool(
    name="eidos_context_suggest",
    description="Get contextually relevant memories based on current activity/prompt.",
    parameters={
        "type": "object",
        "properties": {
            "context": {
                "type": "string",
                "description": "Current activity, prompt, or context to match against"
            },
            "include_self": {
                "type": "boolean",
                "description": "Include EIDOS self-memories (default: true)"
            },
            "include_user": {
                "type": "boolean",
                "description": "Include user memories (default: true)"
            },
            "include_task": {
                "type": "boolean",
                "description": "Include task memories (default: true)"
            },
            "limit": {
                "type": "integer",
                "description": "Max results per category (default: 3)"
            },
        },
        "required": ["context"],
    },
)
@eidosian()
def eidos_context_suggest(
    context: str,
    include_self: bool = True,
    include_user: bool = True,
    include_task: bool = True,
    limit: int = 3,
) -> str:
    """Get contextually relevant memories across categories."""
    mem = _get_tiered_memory()
    if not mem:
        return "Error: Tiered memory system not available"
    
    suggestions = {"context": context, "suggestions": {}}
    
    if include_self:
        self_results = mem.recall_self(context, limit=limit)
        if self_results:
            suggestions["suggestions"]["self"] = [
                {"content": r.content, "tags": list(r.tags)} for r in self_results
            ]
    
    if include_user:
        user_results = mem.recall_user(context, user_id="lloyd", limit=limit)
        if user_results:
            suggestions["suggestions"]["user"] = [
                {"content": r.content, "tags": list(r.tags)} for r in user_results
            ]
    
    if include_task:
        task_results = mem.recall(
            context,
            namespaces=[MemoryNamespace.TASK],
            limit=limit,
        )
        if task_results:
            suggestions["suggestions"]["task"] = [
                {"content": r.content, "tags": list(r.tags)} for r in task_results
            ]
    
    if not suggestions["suggestions"]:
        return "No relevant context found"
    
    return json.dumps(suggestions, indent=2)


# =============================================================================
# AUTO-CONTEXT ENGINE (Advanced context suggestion)
# =============================================================================

_auto_context_engine = None


def _get_auto_context_engine():
    """Lazy-load the auto-context engine."""
    global _auto_context_engine
    if _auto_context_engine is None and TIERED_AVAILABLE:
        try:
            from memory_forge import AutoContextEngine
            mem = _get_tiered_memory()
            if mem:
                _auto_context_engine = AutoContextEngine(mem)
        except ImportError:
            pass
    return _auto_context_engine


@tool(
    name="eidos_auto_context",
    description="Automatically surface relevant memories based on query/prompt. Uses semantic matching, keyword matching, and relevance scoring.",
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Query/prompt to find relevant context for"
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum suggestions to return (default: 5)"
            },
            "min_score": {
                "type": "number",
                "description": "Minimum relevance score threshold (default: 0.3)"
            },
            "format": {
                "type": "string",
                "enum": ["brief", "detailed", "json"],
                "description": "Output format (default: brief)"
            },
        },
        "required": ["query"],
    },
)
@eidosian()
def eidos_auto_context(
    query: str,
    max_results: int = 5,
    min_score: float = 0.3,
    format: str = "brief",
) -> str:
    """Get automatic context suggestions using the AutoContextEngine."""
    engine = _get_auto_context_engine()
    if not engine:
        return "Error: Auto-context engine not available"
    
    suggestions = engine.suggest_context(query, max_suggestions=max_results, min_score=min_score)
    
    if not suggestions:
        return "No relevant context found for query"
    
    return engine.format_suggestions(suggestions, format_type=format)


@tool(
    name="eidos_context_ingest",
    description="Record a prompt, command, or output for context tracking. Helps improve future context suggestions.",
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The content to record"
            },
            "content_type": {
                "type": "string",
                "enum": ["prompt", "command", "output"],
                "description": "Type of content being ingested"
            },
        },
        "required": ["content", "content_type"],
    },
)
@eidosian()
def eidos_context_ingest(content: str, content_type: str) -> str:
    """Ingest content for context tracking."""
    engine = _get_auto_context_engine()
    if not engine:
        return "Error: Auto-context engine not available"
    
    if content_type == "prompt":
        engine.ingest_prompt(content)
    elif content_type == "command":
        engine.ingest_command(content)
    elif content_type == "output":
        engine.ingest_output(content)
    else:
        return f"Error: Unknown content type '{content_type}'"
    
    return f"Context ingested: {content_type}"


# =============================================================================
# MEMORY INTROSPECTION TOOLS
# =============================================================================

_memory_introspector = None


def _get_introspector():
    """Lazy-load the memory introspector."""
    global _memory_introspector
    if _memory_introspector is None:
        try:
            from memory_forge import MemoryIntrospector
            _memory_introspector = MemoryIntrospector()
        except ImportError:
            pass
    return _memory_introspector


@tool(
    name="eidos_memory_introspect",
    description="Analyze memory patterns and generate insights about the memory system.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def eidos_memory_introspect() -> str:
    """Get a comprehensive memory introspection report."""
    introspector = _get_introspector()
    if not introspector:
        return "Error: Memory introspector not available"
    
    return introspector.generate_summary()


@tool(
    name="eidos_memory_stats",
    description="Get detailed statistics about the tiered memory system.",
    parameters={"type": "object", "properties": {}},
)
@eidosian()
def eidos_memory_stats_detailed() -> str:
    """Get detailed memory statistics."""
    import json
    introspector = _get_introspector()
    if not introspector:
        return "Error: Memory introspector not available"
    
    stats = introspector.get_stats()
    return json.dumps({
        "total_memories": stats.total_memories,
        "by_tier": stats.by_tier,
        "by_namespace": stats.by_namespace,
        "by_type": stats.by_type,
        "top_tags": stats.top_tags,
        "avg_importance": round(stats.avg_importance, 2),
        "avg_access_count": round(stats.avg_access_count, 2),
    }, indent=2)


@tool(
    name="eidos_suggest_tags",
    description="Get tag suggestions for a memory based on content analysis.",
    parameters={
        "type": "object",
        "properties": {
            "memory_id": {
                "type": "string",
                "description": "ID of the memory to analyze"
            },
        },
        "required": ["memory_id"],
    },
)
@eidosian()
def eidos_suggest_tags(memory_id: str) -> str:
    """Get tag suggestions for a memory."""
    import json
    introspector = _get_introspector()
    if not introspector:
        return "Error: Memory introspector not available"
    
    suggestions = introspector.suggest_tags(memory_id)
    if not suggestions:
        return "No additional tags suggested for this memory"
    
    return json.dumps({"suggested_tags": suggestions})
