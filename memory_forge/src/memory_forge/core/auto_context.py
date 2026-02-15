"""
Auto-Context Memory Module for EIDOS.

This module provides automatic context surfacing based on:
- Current prompt/query embeddings
- Recent command outputs
- Conversation history
- Active task context

The goal is to proactively suggest relevant memories without explicit recall.
"""
from __future__ import annotations
from eidosian_core import eidosian

import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from .interfaces import MemoryItem
from .tiered_memory import (
    MemoryNamespace,
    MemoryTier,
    TieredMemoryItem,
    TieredMemorySystem,
)

logger = logging.getLogger(__name__)

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()


class EmbedderProtocol(Protocol):
    """Protocol for embedding providers."""
    
    @eidosian()
    def embed(self, text: str) -> List[float]:
        """Embed text into vector."""
        ...


@dataclass
class ContextSuggestion:
    """A single context suggestion with relevance metadata."""
    memory: TieredMemoryItem
    relevance_score: float
    match_reason: str  # Why this was suggested
    source_tier: MemoryTier
    timestamp: datetime = field(default_factory=datetime.now)
    
    @eidosian()
    def to_dict(self) -> Dict[str, Any]:
        """Serialize suggestion."""
        return {
            "memory_id": self.memory.id,
            "content": self.memory.content[:200] + "..." if len(self.memory.content) > 200 else self.memory.content,
            "relevance_score": round(self.relevance_score, 3),
            "match_reason": self.match_reason,
            "tier": self.source_tier.value,
            "namespace": self.memory.namespace.value,
            "tags": list(self.memory.tags),
        }


@dataclass
class ContextWindow:
    """Sliding window of recent context for improved matching."""
    prompts: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)
    commands: List[str] = field(default_factory=list)
    max_items: int = 10
    
    @eidosian()
    def add_prompt(self, prompt: str) -> None:
        """Add a prompt to the window."""
        self.prompts.append(prompt)
        if len(self.prompts) > self.max_items:
            self.prompts.pop(0)
    
    @eidosian()
    def add_output(self, output: str) -> None:
        """Add an output to the window."""
        self.outputs.append(output)
        if len(self.outputs) > self.max_items:
            self.outputs.pop(0)
    
    @eidosian()
    def add_command(self, command: str) -> None:
        """Add a command to the window."""
        self.commands.append(command)
        if len(self.commands) > self.max_items:
            self.commands.pop(0)
    
    @eidosian()
    def get_combined_context(self) -> str:
        """Get all context combined for embedding."""
        parts = []
        if self.prompts:
            parts.append("Recent prompts: " + " | ".join(self.prompts[-3:]))
        if self.commands:
            parts.append("Recent commands: " + " | ".join(self.commands[-3:]))
        return " ".join(parts)


class AutoContextEngine:
    """
    Automatic context suggestion engine.
    
    Monitors prompts, commands, and outputs to proactively surface
    relevant memories from the tiered memory system.
    """
    
    # Minimum score threshold for suggestions
    MIN_RELEVANCE_THRESHOLD = 0.3
    
    # Weights for different matching strategies
    WEIGHTS = {
        "semantic": 0.4,    # Embedding similarity
        "keyword": 0.3,     # Keyword/tag matching
        "recency": 0.15,    # Recent memories preferred
        "importance": 0.15, # High importance memories preferred
    }
    
    # Namespace priorities for different contexts
    NAMESPACE_PRIORITIES = {
        "code": [MemoryNamespace.CODE, MemoryNamespace.KNOWLEDGE, MemoryNamespace.EIDOS],
        "identity": [MemoryNamespace.EIDOS, MemoryNamespace.USER],
        "task": [MemoryNamespace.TASK, MemoryNamespace.CODE, MemoryNamespace.KNOWLEDGE],
        "conversation": [MemoryNamespace.CONVERSATION, MemoryNamespace.USER],
        "default": [ns for ns in MemoryNamespace],
    }
    
    def __init__(
        self,
        memory_system: TieredMemorySystem,
        embedder: Optional[EmbedderProtocol] = None,
    ):
        self.memory = memory_system
        self.embedder = embedder
        self.context_window = ContextWindow()
        self._query_cache: Dict[str, List[float]] = {}
        
    @eidosian()
    def set_embedder(self, embedder: EmbedderProtocol) -> None:
        """Set the embedder for semantic matching."""
        self.embedder = embedder
    
    @eidosian()
    def ingest_prompt(self, prompt: str) -> List[ContextSuggestion]:
        """
        Process a new prompt and return relevant context suggestions.
        
        This is the main entry point - call this when a user sends a prompt.
        """
        self.context_window.add_prompt(prompt)
        return self.suggest_context(prompt)
    
    @eidosian()
    def ingest_command(self, command: str) -> None:
        """Record a command for context tracking."""
        self.context_window.add_command(command)
    
    @eidosian()
    def ingest_output(self, output: str) -> None:
        """Record command output for context tracking."""
        self.context_window.add_output(output)
    
    @eidosian()
    def suggest_context(
        self,
        query: str,
        max_suggestions: int = 5,
        min_score: Optional[float] = None,
    ) -> List[ContextSuggestion]:
        """
        Generate context suggestions for a query.
        
        Args:
            query: The query/prompt to find context for
            max_suggestions: Maximum number of suggestions
            min_score: Minimum relevance score (default: MIN_RELEVANCE_THRESHOLD)
            
        Returns:
            List of ContextSuggestion sorted by relevance
        """
        min_score = min_score or self.MIN_RELEVANCE_THRESHOLD
        
        # Detect query context type
        context_type = self._detect_context_type(query)
        priority_namespaces = self.NAMESPACE_PRIORITIES.get(
            context_type, self.NAMESPACE_PRIORITIES["default"]
        )
        
        # Get all relevant memories
        all_memories = self.memory.list_all()
        
        if not all_memories:
            return []
        
        # Score each memory
        scored: List[Tuple[TieredMemoryItem, float, str]] = []
        
        for item in all_memories:
            score, reason = self._score_memory(query, item, priority_namespaces)
            if score >= min_score:
                scored.append((item, score, reason))
        
        # Sort by score and take top suggestions
        scored.sort(key=lambda x: x[1], reverse=True)
        top_items = scored[:max_suggestions]
        
        # Convert to suggestions
        suggestions = [
            ContextSuggestion(
                memory=item,
                relevance_score=score,
                match_reason=reason,
                source_tier=item.tier,
            )
            for item, score, reason in top_items
        ]
        
        return suggestions
    
    def _detect_context_type(self, query: str) -> str:
        """Detect the type of context being queried."""
        query_lower = query.lower()
        
        # Code-related patterns
        code_patterns = [
            r'\b(function|class|method|variable|import|module)\b',
            r'\b(python|javascript|rust|go|java)\b',
            r'\b(fix|bug|error|test|debug)\b',
            r'\b(code|script|program|library)\b',
        ]
        for pattern in code_patterns:
            if re.search(pattern, query_lower):
                return "code"
        
        # Identity-related patterns
        identity_patterns = [
            r'\b(eidos|identity|who am i|self|myself)\b',
            r'\b(capabilities|lessons|principles)\b',
        ]
        for pattern in identity_patterns:
            if re.search(pattern, query_lower):
                return "identity"
        
        # Task-related patterns
        task_patterns = [
            r'\b(task|todo|plan|implement|create|build)\b',
            r'\b(forge|system|architecture)\b',
        ]
        for pattern in task_patterns:
            if re.search(pattern, query_lower):
                return "task"
        
        return "default"
    
    def _score_memory(
        self,
        query: str,
        item: TieredMemoryItem,
        priority_namespaces: List[MemoryNamespace],
    ) -> Tuple[float, str]:
        """
        Calculate relevance score for a memory item.
        
        Returns:
            Tuple of (score, reason string)
        """
        scores: Dict[str, float] = {}
        reasons: List[str] = []
        
        # 1. Semantic similarity (if embedder available)
        if self.embedder and item.embedding:
            query_embedding = self._get_embedding(query)
            if query_embedding:
                semantic_sim = self._cosine_similarity(query_embedding, item.embedding)
                scores["semantic"] = semantic_sim
                if semantic_sim > 0.5:
                    reasons.append(f"semantic:{semantic_sim:.2f}")
        
        # 2. Keyword matching
        keyword_score = self._keyword_match_score(query, item)
        scores["keyword"] = keyword_score
        if keyword_score > 0.3:
            reasons.append(f"keyword:{keyword_score:.2f}")
        
        # 3. Recency score (normalized to 0-1 based on age)
        age_hours = (datetime.now() - item.last_accessed).total_seconds() / 3600
        recency_score = max(0, 1.0 - (age_hours / 168))  # Decay over 1 week
        scores["recency"] = recency_score
        
        # 4. Importance score (already 0-1)
        scores["importance"] = min(1.0, item.importance)
        
        # 5. Namespace priority bonus
        namespace_bonus = 0.0
        if item.namespace in priority_namespaces:
            idx = priority_namespaces.index(item.namespace)
            namespace_bonus = 0.2 * (1.0 - idx / len(priority_namespaces))
        
        # 6. Tier bonus (SELF and USER memories get slight boost)
        tier_bonus = 0.0
        if item.tier == MemoryTier.SELF:
            tier_bonus = 0.15
        elif item.tier == MemoryTier.USER:
            tier_bonus = 0.1
        elif item.tier == MemoryTier.LONG_TERM:
            tier_bonus = 0.05
        
        # Calculate weighted score
        final_score = sum(
            scores.get(key, 0) * weight
            for key, weight in self.WEIGHTS.items()
        ) + namespace_bonus + tier_bonus
        
        # Compile reason string
        if not reasons:
            reasons = [f"base:{final_score:.2f}"]
        
        return final_score, ", ".join(reasons)
    
    def _keyword_match_score(self, query: str, item: TieredMemoryItem) -> float:
        """Calculate keyword match score."""
        query_words = set(query.lower().split())
        content_words = set(item.content.lower().split())
        
        # Direct overlap
        overlap = len(query_words & content_words)
        overlap_score = min(1.0, overlap / max(len(query_words), 1))
        
        # Tag matches
        tag_matches = sum(1 for tag in item.tags if tag.lower() in query.lower())
        tag_score = min(1.0, tag_matches * 0.3)
        
        # Exact phrase match bonus
        phrase_bonus = 0.3 if query.lower() in item.content.lower() else 0.0
        
        return min(1.0, overlap_score * 0.5 + tag_score * 0.3 + phrase_bonus * 0.2)
    
    def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text, with caching."""
        if not self.embedder:
            return None
        
        # Use cache for repeated queries
        cache_key = text[:100]  # Truncate for cache key
        if cache_key in self._query_cache:
            return self._query_cache[cache_key]
        
        try:
            embedding = self.embedder.embed(text)
            self._query_cache[cache_key] = embedding
            # Limit cache size
            if len(self._query_cache) > 100:
                # Remove oldest entries
                keys_to_remove = list(self._query_cache.keys())[:50]
                for k in keys_to_remove:
                    del self._query_cache[k]
            return embedding
        except Exception as e:
            logger.warning(f"Embedding failed: {e}")
            return None
    
    def _cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity."""
        if not vec_a or not vec_b or len(vec_a) != len(vec_b):
            return 0.0
        dot = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = sum(a * a for a in vec_a) ** 0.5
        norm_b = sum(b * b for b in vec_b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
    
    @eidosian()
    def format_suggestions(
        self,
        suggestions: List[ContextSuggestion],
        format_type: str = "brief",
    ) -> str:
        """
        Format suggestions for display.
        
        Args:
            suggestions: List of context suggestions
            format_type: "brief" | "detailed" | "json"
        """
        if not suggestions:
            return "No relevant context found."
        
        if format_type == "json":
            import json
            return json.dumps([s.to_dict() for s in suggestions], indent=2)
        
        lines = [f"📚 Found {len(suggestions)} relevant context items:"]
        
        for i, sugg in enumerate(suggestions, 1):
            content_preview = (
                sugg.memory.content[:100] + "..."
                if len(sugg.memory.content) > 100
                else sugg.memory.content
            )
            
            if format_type == "brief":
                lines.append(
                    f"  {i}. [{sugg.source_tier.value}] "
                    f"(score:{sugg.relevance_score:.2f}) {content_preview}"
                )
            else:  # detailed
                lines.append(f"\n{i}. Memory [{sugg.memory.id[:8]}]")
                lines.append(f"   Tier: {sugg.source_tier.value}")
                lines.append(f"   Namespace: {sugg.memory.namespace.value}")
                lines.append(f"   Score: {sugg.relevance_score:.3f}")
                lines.append(f"   Reason: {sugg.match_reason}")
                lines.append(f"   Tags: {', '.join(sugg.memory.tags) or 'none'}")
                lines.append(f"   Content: {content_preview}")
        
        return "\n".join(lines)


# Convenience function for quick context lookup
@eidosian()
def get_auto_context(
    prompt: str,
    persistence_dir: str | None = None,
    max_suggestions: int = 5,
) -> List[ContextSuggestion]:
    """
    Quick function to get context suggestions for a prompt.
    
    Usage:
        suggestions = get_auto_context("How do I fix the word_forge imports?")
        for s in suggestions:
            print(f"- {s.memory.content[:100]}... (score: {s.relevance_score})")
    """
    memory_path = Path(persistence_dir) if persistence_dir else (FORGE_ROOT / "data" / "memory")
    memory_system = TieredMemorySystem(persistence_dir=memory_path)
    engine = AutoContextEngine(memory_system)
    return engine.suggest_context(prompt, max_suggestions=max_suggestions)
