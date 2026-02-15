"""
Knowledge-Memory Bridge for EIDOS.

This module connects knowledge_forge and memory_forge to enable:
- Bi-directional cross-referencing between memories and knowledge nodes
- Unified search across both systems
- Automatic knowledge node creation from important memories
- Memory enrichment with relevant knowledge context
"""
from __future__ import annotations
from eidosian_core import eidosian

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[4]))).resolve()

# Default paths
DEFAULT_MEMORY_DIR = FORGE_ROOT / "data" / "memory"
DEFAULT_KB_PATH = FORGE_ROOT / "data" / "kb.json"


@dataclass
class UnifiedSearchResult:
    """A search result from either memory or knowledge system."""
    source: str  # "memory" or "knowledge"
    id: str
    content: str
    score: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    linked_ids: List[str] = field(default_factory=list)


class KnowledgeMemoryBridge:
    """
    Bridge between knowledge_forge and memory_forge.
    
    Enables:
    - Unified search across both systems
    - Automatic cross-linking
    - Memory → Knowledge promotion
    - Knowledge → Memory enrichment
    """
    
    def __init__(
        self,
        memory_dir: Optional[Path] = None,
        kb_path: Optional[Path] = None,
    ):
        self.memory_dir = memory_dir or DEFAULT_MEMORY_DIR
        self.kb_path = kb_path or DEFAULT_KB_PATH
        
        self._memory_system = None
        self._knowledge_forge = None
        
        # Cross-reference maps
        self.memory_to_knowledge: Dict[str, Set[str]] = {}  # memory_id -> knowledge_ids
        self.knowledge_to_memory: Dict[str, Set[str]] = {}  # knowledge_id -> memory_ids
        
        self._load_xref()
    
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
    
    @property
    def knowledge(self):
        """Lazy-load knowledge forge."""
        if self._knowledge_forge is None:
            try:
                from knowledge_forge import KnowledgeForge
                self._knowledge_forge = KnowledgeForge(persistence_path=self.kb_path)
            except ImportError as e:
                logger.warning(f"Could not import KnowledgeForge: {e}")
        return self._knowledge_forge
    
    @eidosian()
    def unified_search(
        self,
        query: str,
        include_memory: bool = True,
        include_knowledge: bool = True,
        limit: int = 10,
    ) -> List[UnifiedSearchResult]:
        """
        Search across both memory and knowledge systems.
        
        Returns results sorted by relevance score.
        """
        results: List[UnifiedSearchResult] = []
        
        # Search memories
        if include_memory and self.memory:
            try:
                memories = self.memory.recall(query, limit=limit)
                for mem in memories:
                    # Calculate simple score based on content match
                    score = self._calculate_score(query, mem.content)
                    results.append(UnifiedSearchResult(
                        source="memory",
                        id=mem.id,
                        content=mem.content,
                        score=score,
                        metadata={
                            "tier": mem.tier.value,
                            "namespace": mem.namespace.value,
                            "tags": list(mem.tags),
                            "importance": mem.importance,
                        },
                        linked_ids=list(self.memory_to_knowledge.get(mem.id, set())),
                    ))
            except Exception as e:
                logger.warning(f"Memory search failed: {e}")
        
        # Search knowledge
        if include_knowledge and self.knowledge:
            try:
                nodes = self.knowledge.search(query)
                for node in nodes[:limit]:
                    score = self._calculate_score(query, str(node.content))
                    results.append(UnifiedSearchResult(
                        source="knowledge",
                        id=node.id,
                        content=str(node.content),
                        score=score,
                        metadata={
                            "tags": list(node.tags),
                            **node.metadata,
                        },
                        linked_ids=list(self.knowledge_to_memory.get(node.id, set())),
                    ))
            except Exception as e:
                logger.warning(f"Knowledge search failed: {e}")
        
        # Sort by score
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]
    
    @eidosian()
    def link_memory_to_knowledge(
        self,
        memory_id: str,
        knowledge_id: str,
    ) -> bool:
        """Create a cross-reference link between memory and knowledge node."""
        # Update memory → knowledge map
        if memory_id not in self.memory_to_knowledge:
            self.memory_to_knowledge[memory_id] = set()
        self.memory_to_knowledge[memory_id].add(knowledge_id)
        
        # Update knowledge → memory map
        if knowledge_id not in self.knowledge_to_memory:
            self.knowledge_to_memory[knowledge_id] = set()
        self.knowledge_to_memory[knowledge_id].add(memory_id)
        
        self._save_xref()
        return True
    
    @eidosian()
    def promote_memory_to_knowledge(
        self,
        memory_id: str,
        concepts: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Promote a memory to a knowledge node.
        
        This is useful for important memories that should become
        permanent knowledge entries.
        
        Returns the new knowledge node ID, or None if failed.
        """
        if not self.memory or not self.knowledge:
            return None
        
        # Find the memory
        mem = self.memory._find_memory(memory_id)
        if not mem:
            logger.warning(f"Memory {memory_id} not found")
            return None
        
        # Create knowledge node from memory
        metadata = {
            "source": "memory_promotion",
            "original_memory_id": memory_id,
            "promoted_at": datetime.now().isoformat(),
            "original_tier": mem.tier.value,
            "original_namespace": mem.namespace.value,
        }
        metadata.update(mem.metadata)
        
        # Merge tags
        all_tags = list(mem.tags)
        if tags:
            all_tags.extend(tags)
        
        node = self.knowledge.add_knowledge(
            content=mem.content,
            concepts=concepts,
            tags=all_tags,
            metadata=metadata,
        )
        
        # Create cross-reference
        self.link_memory_to_knowledge(memory_id, node.id)
        
        return node.id
    
    @eidosian()
    def enrich_memory_context(
        self,
        memory_id: str,
        max_knowledge: int = 3,
    ) -> List[Dict[str, Any]]:
        """
        Find relevant knowledge nodes for a memory.
        
        Returns list of knowledge nodes that are semantically
        related to the memory content.
        """
        if not self.memory or not self.knowledge:
            return []
        
        # Get the memory
        mem = self.memory._find_memory(memory_id)
        if not mem:
            return []
        
        # Search knowledge for related content
        related = self.knowledge.search(mem.content)
        
        results = []
        for node in related[:max_knowledge]:
            results.append({
                "knowledge_id": node.id,
                "content": str(node.content)[:200],
                "tags": list(node.tags),
            })
        
        return results
    
    @eidosian()
    def get_memory_knowledge_context(
        self,
        query: str,
        max_results: int = 5,
    ) -> Dict[str, Any]:
        """
        Get comprehensive context from both systems for a query.
        
        This is the primary method for autonomous context suggestion.
        """
        results = self.unified_search(query, limit=max_results)
        
        # Organize by source
        memory_context = [r for r in results if r.source == "memory"]
        knowledge_context = [r for r in results if r.source == "knowledge"]
        
        return {
            "query": query,
            "total_results": len(results),
            "memory_context": [
                {
                    "id": r.id,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "tier": r.metadata.get("tier"),
                    "namespace": r.metadata.get("namespace"),
                }
                for r in memory_context
            ],
            "knowledge_context": [
                {
                    "id": r.id,
                    "content": r.content[:200] + "..." if len(r.content) > 200 else r.content,
                    "score": r.score,
                    "tags": r.metadata.get("tags", []),
                }
                for r in knowledge_context
            ],
        }
    
    @eidosian()
    def stats(self) -> Dict[str, Any]:
        """Get statistics about the bridge."""
        memory_count = 0
        knowledge_count = 0
        
        if self.memory:
            try:
                memory_count = self.memory.stats()["total"]
            except:
                pass
        
        if self.knowledge:
            try:
                knowledge_count = self.knowledge.stats()["node_count"]
            except:
                pass
        
        return {
            "memory_count": memory_count,
            "knowledge_count": knowledge_count,
            "memory_to_knowledge_links": sum(len(v) for v in self.memory_to_knowledge.values()),
            "knowledge_to_memory_links": sum(len(v) for v in self.knowledge_to_memory.values()),
            "linked_memories": len(self.memory_to_knowledge),
            "linked_knowledge_nodes": len(self.knowledge_to_memory),
        }
    
    def _calculate_score(self, query: str, content: str) -> float:
        """Calculate simple relevance score."""
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Exact phrase match
        if query_lower in content_lower:
            return 1.0
        
        # Word overlap
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())
        overlap = len(query_words & content_words)
        
        if not query_words:
            return 0.0
        
        return min(1.0, overlap / len(query_words))
    
    def _load_xref(self) -> None:
        """Load cross-reference maps from disk."""
        xref_path = self.memory_dir / "knowledge_xref.json"
        if xref_path.exists():
            try:
                with open(xref_path) as f:
                    data = json.load(f)
                self.memory_to_knowledge = {
                    k: set(v) for k, v in data.get("memory_to_knowledge", {}).items()
                }
                self.knowledge_to_memory = {
                    k: set(v) for k, v in data.get("knowledge_to_memory", {}).items()
                }
            except Exception as e:
                logger.warning(f"Failed to load xref: {e}")
    
    def _save_xref(self) -> None:
        """Save cross-reference maps to disk."""
        xref_path = self.memory_dir / "knowledge_xref.json"
        try:
            data = {
                "memory_to_knowledge": {
                    k: list(v) for k, v in self.memory_to_knowledge.items()
                },
                "knowledge_to_memory": {
                    k: list(v) for k, v in self.knowledge_to_memory.items()
                },
            }
            with open(xref_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save xref: {e}")


# Convenience function
@eidosian()
def get_unified_context(
    query: str,
    max_results: int = 5,
) -> Dict[str, Any]:
    """Quick function to get unified context for a query."""
    bridge = KnowledgeMemoryBridge()
    return bridge.get_memory_knowledge_context(query, max_results=max_results)
