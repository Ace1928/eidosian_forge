"""
Memory Introspection Module for EIDOS.

Analyzes memory patterns, generates insights, and provides recommendations
for memory organization and evolution.
"""
from __future__ import annotations
from eidosian_core import eidosian

import json
import logging
import os
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()


@dataclass
class MemoryInsight:
    """An insight derived from memory analysis."""
    insight_type: str  # "pattern", "gap", "recommendation", "trend"
    description: str
    confidence: float  # 0-1
    evidence: List[str]  # memory IDs supporting this insight
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class MemoryStats:
    """Statistics about the memory system."""
    total_memories: int = 0
    by_tier: Dict[str, int] = field(default_factory=dict)
    by_namespace: Dict[str, int] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    top_tags: List[Tuple[str, int]] = field(default_factory=list)
    avg_importance: float = 0.0
    avg_access_count: float = 0.0
    oldest_memory: Optional[datetime] = None
    newest_memory: Optional[datetime] = None


class MemoryIntrospector:
    """
    Analyzes memory patterns and generates insights.
    
    Capabilities:
    - Memory pattern detection
    - Gap analysis (what's missing)
    - Trend identification
    - Auto-tagging suggestions
    - Organization recommendations
    """
    
    def __init__(self, memory_dir: Optional[Path] = None):
        self.memory_dir = memory_dir or (FORGE_ROOT / "data" / "memory")
        self._memory_system = None
    
    @property
    def memory(self):
        """Lazy-load memory system."""
        if self._memory_system is None:
            try:
                from memory_forge import TieredMemorySystem
                self._memory_system = TieredMemorySystem(persistence_dir=self.memory_dir)
            except ImportError as e:
                logger.warning(f"Could not import TieredMemorySystem: {e}")
        return self._memory_system
    
    @eidosian()
    def get_stats(self) -> MemoryStats:
        """Get comprehensive statistics about the memory system."""
        if not self.memory:
            return MemoryStats()
        
        stats = MemoryStats()
        all_memories = self.memory.list_all()
        stats.total_memories = len(all_memories)
        
        if not all_memories:
            return stats
        
        # Count by tier, namespace, type
        tier_counts: Dict[str, int] = {}
        namespace_counts: Dict[str, int] = {}
        type_counts: Dict[str, int] = {}
        tag_counter: Counter = Counter()
        total_importance = 0.0
        total_access = 0
        oldest = datetime.max
        newest = datetime.min
        
        for mem in all_memories:
            # Tier
            tier = mem.tier.value
            tier_counts[tier] = tier_counts.get(tier, 0) + 1
            
            # Namespace
            ns = mem.namespace.value
            namespace_counts[ns] = namespace_counts.get(ns, 0) + 1
            
            # Type
            mt = mem.memory_type.value
            type_counts[mt] = type_counts.get(mt, 0) + 1
            
            # Tags
            for tag in mem.tags:
                tag_counter[tag] += 1
            
            # Stats
            total_importance += mem.importance
            total_access += mem.access_count
            
            if mem.created_at < oldest:
                oldest = mem.created_at
            if mem.created_at > newest:
                newest = mem.created_at
        
        stats.by_tier = tier_counts
        stats.by_namespace = namespace_counts
        stats.by_type = type_counts
        stats.top_tags = tag_counter.most_common(10)
        stats.avg_importance = total_importance / len(all_memories)
        stats.avg_access_count = total_access / len(all_memories)
        stats.oldest_memory = oldest if oldest != datetime.max else None
        stats.newest_memory = newest if newest != datetime.min else None
        
        return stats
    
    @eidosian()
    def analyze_patterns(self) -> List[MemoryInsight]:
        """Detect patterns in memory content and organization."""
        if not self.memory:
            return []
        
        insights: List[MemoryInsight] = []
        all_memories = self.memory.list_all()
        
        if len(all_memories) < 5:
            return insights
        
        # Pattern 1: Tier imbalance
        stats = self.get_stats()
        if stats.by_tier:
            total = sum(stats.by_tier.values())
            for tier, count in stats.by_tier.items():
                ratio = count / total
                if tier == "self" and ratio > 0.5:
                    insights.append(MemoryInsight(
                        insight_type="pattern",
                        description=f"High concentration of SELF memories ({ratio:.0%}). Consider organizing into more specific categories.",
                        confidence=0.8,
                        evidence=[],
                    ))
                elif tier == "short_term" and count > 50:
                    insights.append(MemoryInsight(
                        insight_type="recommendation",
                        description=f"Large number of SHORT_TERM memories ({count}). Consider promoting important ones to WORKING or LONG_TERM.",
                        confidence=0.7,
                        evidence=[],
                    ))
        
        # Pattern 2: Untagged memories
        untagged = [m for m in all_memories if not m.tags]
        if len(untagged) > len(all_memories) * 0.3:
            insights.append(MemoryInsight(
                insight_type="gap",
                description=f"{len(untagged)} memories lack tags ({len(untagged)/len(all_memories):.0%}). Tags improve retrieval.",
                confidence=0.9,
                evidence=[m.id for m in untagged[:5]],
            ))
        
        # Pattern 3: Low importance memories
        low_importance = [m for m in all_memories if m.importance < 0.3]
        if len(low_importance) > 20:
            insights.append(MemoryInsight(
                insight_type="recommendation",
                description=f"{len(low_importance)} memories have low importance (<0.3). Consider cleanup or importance adjustment.",
                confidence=0.6,
                evidence=[m.id for m in low_importance[:5]],
            ))
        
        # Pattern 4: Highly accessed memories
        high_access = [m for m in all_memories if m.access_count > 5]
        if high_access:
            insights.append(MemoryInsight(
                insight_type="trend",
                description=f"{len(high_access)} memories are frequently accessed. These are core knowledge.",
                confidence=0.8,
                evidence=[m.id for m in high_access[:5]],
                metadata={"avg_access": sum(m.access_count for m in high_access) / len(high_access)},
            ))
        
        # Pattern 5: Namespace gaps
        expected_namespaces = {"eidos", "user", "task", "knowledge", "code", "conversation"}
        existing = set(stats.by_namespace.keys())
        missing = expected_namespaces - existing
        if missing:
            insights.append(MemoryInsight(
                insight_type="gap",
                description=f"No memories in namespaces: {', '.join(missing)}. Consider populating these.",
                confidence=0.5,
                evidence=[],
            ))
        
        return insights
    
    @eidosian()
    def suggest_tags(self, memory_id: str) -> List[str]:
        """Suggest tags for a memory based on content analysis."""
        if not self.memory:
            return []
        
        mem = self.memory._find_memory(memory_id)
        if not mem:
            return []
        
        suggestions: Set[str] = set()
        content_lower = mem.content.lower()
        
        # Keyword-based suggestions
        keyword_tags = {
            "lesson": ["lesson", "learned", "insight", "mistake", "error"],
            "identity": ["identity", "eidos", "who am i", "self"],
            "code": ["function", "class", "module", "python", "import"],
            "task": ["task", "todo", "implement", "create", "build"],
            "architecture": ["architecture", "design", "structure", "system"],
            "memory": ["memory", "recall", "remember", "store"],
            "introspection": ["introspection", "reflect", "analyze"],
        }
        
        for tag, keywords in keyword_tags.items():
            if any(kw in content_lower for kw in keywords):
                suggestions.add(tag)
        
        # Tier-based suggestions
        if mem.tier.value == "self":
            suggestions.add("eidos")
        elif mem.tier.value == "user":
            suggestions.add("user_preference")
        
        return list(suggestions - mem.tags)  # Only new suggestions
    
    @eidosian()
    def get_organization_recommendations(self) -> List[str]:
        """Get recommendations for better memory organization."""
        recommendations: List[str] = []
        stats = self.get_stats()
        
        if stats.total_memories > 100:
            recommendations.append(
                "Consider implementing periodic memory consolidation to merge related memories."
            )
        
        if stats.avg_importance > 0.7:
            recommendations.append(
                "Many high-importance memories. Consider reviewing importance scores for accuracy."
            )
        
        if stats.by_type.get("episodic", 0) > 50 and stats.by_type.get("semantic", 0) < 10:
            recommendations.append(
                "Heavy episodic vs semantic imbalance. Extract key facts into semantic memories."
            )
        
        if not stats.by_namespace.get("code"):
            recommendations.append(
                "No code-related memories. Consider storing lessons from coding sessions."
            )
        
        return recommendations
    
    @eidosian()
    def generate_summary(self) -> str:
        """Generate a comprehensive memory system summary."""
        stats = self.get_stats()
        insights = self.analyze_patterns()
        recommendations = self.get_organization_recommendations()
        
        lines = [
            "# Memory System Introspection Report",
            f"Generated: {datetime.now().isoformat()}",
            "",
            "## Statistics",
            f"- Total memories: {stats.total_memories}",
            f"- Average importance: {stats.avg_importance:.2f}",
            f"- Average access count: {stats.avg_access_count:.1f}",
        ]
        
        if stats.oldest_memory:
            lines.append(f"- Date range: {stats.oldest_memory.date()} to {stats.newest_memory.date()}")
        
        lines.append("\n### By Tier")
        for tier, count in sorted(stats.by_tier.items()):
            lines.append(f"  - {tier}: {count}")
        
        lines.append("\n### By Namespace")
        for ns, count in sorted(stats.by_namespace.items()):
            lines.append(f"  - {ns}: {count}")
        
        if stats.top_tags:
            lines.append("\n### Top Tags")
            for tag, count in stats.top_tags[:5]:
                lines.append(f"  - {tag}: {count}")
        
        if insights:
            lines.append("\n## Insights")
            for insight in insights:
                lines.append(f"- [{insight.insight_type.upper()}] {insight.description}")
        
        if recommendations:
            lines.append("\n## Recommendations")
            for rec in recommendations:
                lines.append(f"- {rec}")
        
        return "\n".join(lines)


# Convenience function
@eidosian()
def introspect_memory() -> str:
    """Quick function to get memory introspection summary."""
    introspector = MemoryIntrospector()
    return introspector.generate_summary()
