#!/usr/bin/env python3
"""
Memory Forge CLI - Command-line interface for the tiered memory system.

Standalone Usage:
    memory-forge status              # Check memory system status
    memory-forge list                # List all memories
    memory-forge search <query>      # Search memories
    memory-forge store <content>     # Store a new memory
    memory-forge introspect          # Analyze memory patterns
    memory-forge context <prompt>    # Get auto-context for a prompt

Enhanced with other forges:
    - knowledge_forge: Unified search across memory and knowledge
    - llm_forge: Smart memory summarization
    - word_forge: Semantic memory relationships
"""
from __future__ import annotations
from eidosian_core import eidosian

import sys
from pathlib import Path
from typing import List, Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector, create_cli_entry_point

from memory_forge import (
    TieredMemorySystem,
    MemoryTier,
    MemoryNamespace,
    MemoryIntrospector,
    AutoContextEngine,
    introspect_memory,
)
from memory_forge.core.interfaces import MemoryType

# Default paths
DEFAULT_MEMORY_DIR = Path("/home/lloyd/eidosian_forge/data/memory")


class MemoryForgeCLI(StandardCLI):
    """CLI for the memory forge tiered memory system."""
    
    name = "memory_forge"
    description = "Tiered memory system for EIDOS - stores, retrieves, and analyzes memories"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._memory: Optional[TieredMemorySystem] = None
        self._introspector: Optional[MemoryIntrospector] = None
        self._auto_context: Optional[AutoContextEngine] = None
    
    @property
    def memory(self) -> TieredMemorySystem:
        """Lazy-load memory system."""
        if self._memory is None:
            self._memory = TieredMemorySystem(persistence_dir=DEFAULT_MEMORY_DIR)
        return self._memory
    
    @property
    def introspector(self) -> MemoryIntrospector:
        """Lazy-load introspector."""
        if self._introspector is None:
            self._introspector = MemoryIntrospector(memory_dir=DEFAULT_MEMORY_DIR)
        return self._introspector
    
    @property
    def auto_context(self) -> AutoContextEngine:
        """Lazy-load auto-context engine."""
        if self._auto_context is None:
            self._auto_context = AutoContextEngine(self.memory)
        return self._auto_context
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register memory-specific commands."""
        
        # List command
        list_parser = subparsers.add_parser(
            "list",
            help="List memories",
        )
        list_parser.add_argument(
            "-t", "--tier",
            choices=["short_term", "working", "long_term", "self", "user"],
            help="Filter by tier",
        )
        list_parser.add_argument(
            "-n", "--namespace",
            choices=["eidos", "user", "task", "knowledge", "code", "conversation"],
            help="Filter by namespace",
        )
        list_parser.add_argument(
            "-l", "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        list_parser.set_defaults(func=self._cmd_list)
        
        # Search command
        search_parser = subparsers.add_parser(
            "search",
            help="Search memories",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )
        search_parser.add_argument(
            "-l", "--limit",
            type=int,
            default=5,
            help="Maximum results (default: 5)",
        )
        search_parser.set_defaults(func=self._cmd_search)
        
        # Store command
        store_parser = subparsers.add_parser(
            "store",
            help="Store a new memory",
        )
        store_parser.add_argument(
            "content",
            help="Memory content",
        )
        store_parser.add_argument(
            "-t", "--tier",
            choices=["short_term", "working", "long_term", "self", "user"],
            default="working",
            help="Memory tier (default: working)",
        )
        store_parser.add_argument(
            "-n", "--namespace",
            choices=["eidos", "user", "task", "knowledge", "code", "conversation"],
            default="task",
            help="Namespace (default: task)",
        )
        store_parser.add_argument(
            "--tags",
            nargs="*",
            help="Tags for the memory",
        )
        store_parser.add_argument(
            "-i", "--importance",
            type=float,
            default=0.5,
            help="Importance score 0-1 (default: 0.5)",
        )
        store_parser.set_defaults(func=self._cmd_store)
        
        # Introspect command
        intro_parser = subparsers.add_parser(
            "introspect",
            help="Analyze memory patterns and get insights",
        )
        intro_parser.set_defaults(func=self._cmd_introspect)
        
        # Context command
        context_parser = subparsers.add_parser(
            "context",
            help="Get automatic context suggestions for a prompt",
        )
        context_parser.add_argument(
            "prompt",
            help="Prompt to find context for",
        )
        context_parser.add_argument(
            "-l", "--limit",
            type=int,
            default=5,
            help="Maximum suggestions (default: 5)",
        )
        context_parser.set_defaults(func=self._cmd_context)
        
        # Stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Get memory statistics",
        )
        stats_parser.set_defaults(func=self._cmd_stats)
        
        # Cleanup command
        cleanup_parser = subparsers.add_parser(
            "cleanup",
            help="Clean up expired memories",
        )
        cleanup_parser.set_defaults(func=self._cmd_cleanup)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Check memory system status."""
        try:
            stats = self.memory.stats()
            return CommandResult(
                True,
                f"Memory system operational - {stats['total']} memories",
                {
                    "total": stats["total"],
                    "by_tier": stats["by_tier"],
                    "by_namespace": stats["by_namespace"],
                    "persistence_dir": str(DEFAULT_MEMORY_DIR),
                },
            )
        except Exception as e:
            return CommandResult(False, f"Memory system error: {e}")
    
    def _cmd_list(self, args) -> None:
        """List memories."""
        tier_filter = None
        if args.tier:
            tier_map = {
                "short_term": MemoryTier.SHORT_TERM,
                "working": MemoryTier.WORKING,
                "long_term": MemoryTier.LONG_TERM,
                "self": MemoryTier.SELF,
                "user": MemoryTier.USER,
            }
            tier_filter = [tier_map[args.tier]]
        
        ns_filter = None
        if args.namespace:
            ns_map = {
                "eidos": MemoryNamespace.EIDOS,
                "user": MemoryNamespace.USER,
                "task": MemoryNamespace.TASK,
                "knowledge": MemoryNamespace.KNOWLEDGE,
                "code": MemoryNamespace.CODE,
                "conversation": MemoryNamespace.CONVERSATION,
            }
            ns_filter = [ns_map[args.namespace]]
        
        memories = self.memory.list_all(tiers=tier_filter, namespaces=ns_filter)
        memories = memories[:args.limit]
        
        data = []
        for mem in memories:
            data.append({
                "id": mem.id[:8],
                "tier": mem.tier.value,
                "namespace": mem.namespace.value,
                "content": mem.content[:60] + "..." if len(mem.content) > 60 else mem.content,
                "tags": list(mem.tags)[:3],
            })
        
        result = CommandResult(
            True,
            f"Found {len(memories)} memories",
            data,
        )
        self._output(result, args)
    
    def _cmd_search(self, args) -> None:
        """Search memories."""
        results = self.memory.recall(args.query, limit=args.limit)
        
        data = []
        for mem in results:
            data.append({
                "id": mem.id[:8],
                "tier": mem.tier.value,
                "content": mem.content[:100] + "..." if len(mem.content) > 100 else mem.content,
            })
        
        result = CommandResult(
            True,
            f"Found {len(results)} memories matching '{args.query}'",
            data,
        )
        self._output(result, args)
    
    def _cmd_store(self, args) -> None:
        """Store a new memory."""
        tier_map = {
            "short_term": MemoryTier.SHORT_TERM,
            "working": MemoryTier.WORKING,
            "long_term": MemoryTier.LONG_TERM,
            "self": MemoryTier.SELF,
            "user": MemoryTier.USER,
        }
        ns_map = {
            "eidos": MemoryNamespace.EIDOS,
            "user": MemoryNamespace.USER,
            "task": MemoryNamespace.TASK,
            "knowledge": MemoryNamespace.KNOWLEDGE,
            "code": MemoryNamespace.CODE,
            "conversation": MemoryNamespace.CONVERSATION,
        }
        
        tags = set(args.tags) if args.tags else set()
        
        memory_id = self.memory.remember(
            content=args.content,
            tier=tier_map[args.tier],
            namespace=ns_map[args.namespace],
            memory_type=MemoryType.EPISODIC,
            tags=tags,
            importance=args.importance,
        )
        
        self.memory.save_all()
        
        result = CommandResult(
            True,
            f"Memory stored: {memory_id[:8]}",
            {"id": memory_id, "tier": args.tier, "namespace": args.namespace},
        )
        self._output(result, args)
    
    def _cmd_introspect(self, args) -> None:
        """Run memory introspection."""
        report = introspect_memory()
        
        result = CommandResult(
            True,
            "Memory introspection complete",
            {"report": report},
        )
        if args.json:
            self._output(result, args)
        else:
            print(report)
    
    def _cmd_context(self, args) -> None:
        """Get auto-context suggestions."""
        suggestions = self.auto_context.suggest_context(args.prompt, max_suggestions=args.limit)
        formatted = self.auto_context.format_suggestions(suggestions, format_type="brief")
        
        data = [
            {
                "tier": s.source_tier.value,
                "score": round(s.relevance_score, 2),
                "content": s.memory.content[:80] + "..." if len(s.memory.content) > 80 else s.memory.content,
            }
            for s in suggestions
        ]
        
        result = CommandResult(
            True,
            f"Found {len(suggestions)} context suggestions for '{args.prompt}'",
            data,
        )
        self._output(result, args)
    
    def _cmd_stats(self, args) -> None:
        """Get detailed statistics."""
        stats = self.introspector.get_stats()
        
        result = CommandResult(
            True,
            f"Memory statistics - {stats.total_memories} total",
            {
                "total_memories": stats.total_memories,
                "by_tier": stats.by_tier,
                "by_namespace": stats.by_namespace,
                "top_tags": stats.top_tags[:5],
                "avg_importance": round(stats.avg_importance, 2),
            },
        )
        self._output(result, args)
    
    def _cmd_cleanup(self, args) -> None:
        """Clean up expired memories."""
        before = self.memory.stats()["total"]
        removed = self.memory.cleanup_expired()
        after = self.memory.stats()["total"]
        
        result = CommandResult(
            True,
            f"Cleanup complete - removed {removed} expired memories",
            {"before": before, "after": after, "removed": removed},
        )
        self._output(result, args)
    
    @eidosian()
    def get_enhanced_capabilities(self, available_forges: List[str]) -> List[str]:
        """Return enhanced capabilities when other forges present."""
        enhanced = []
        if "knowledge_forge" in available_forges:
            enhanced.append("Unified memory+knowledge search via 'context' command")
        if "llm_forge" in available_forges:
            enhanced.append("Smart memory summarization and extraction")
        if "word_forge" in available_forges:
            enhanced.append("Semantic relationship mapping between memories")
        return enhanced


# Entry point
main = create_cli_entry_point(MemoryForgeCLI)

if __name__ == "__main__":
    main()
