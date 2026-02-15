#!/usr/bin/env python3
"""
Knowledge Forge CLI - Command line interface for knowledge graph management.

Standalone Usage:
    knowledge-forge status          # Show knowledge graph status
    knowledge-forge list            # List nodes
    knowledge-forge search <query>  # Search knowledge
    knowledge-forge add <content>   # Add knowledge node
    knowledge-forge link A B        # Link two nodes
    knowledge-forge path A B        # Find path between nodes

Enhanced with other forges:
    - memory_forge: Unified search across memory and knowledge
    - graphrag: Advanced semantic indexing and querying
"""
from __future__ import annotations
from eidosian_core import eidosian

import os
import sys
from pathlib import Path
from typing import Optional

# Add lib to path for CLI framework
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "lib"))

from cli import StandardCLI, CommandResult, ForgeDetector

from knowledge_forge import KnowledgeForge, KnowledgeMemoryBridge

# Default paths
FORGE_ROOT = Path(os.environ.get("EIDOS_FORGE_DIR", str(Path(__file__).resolve().parents[3]))).resolve()
DEFAULT_KB_PATH = FORGE_ROOT / "data" / "kb.json"


class KnowledgeForgeCLI(StandardCLI):
    """CLI for the knowledge graph system."""
    
    name = "knowledge_forge"
    description = "Knowledge graph with concept mapping, tagging, and pathfinding"
    version = "1.0.0"
    
    def __init__(self):
        super().__init__()
        self._kb: Optional[KnowledgeForge] = None
        self._bridge: Optional[KnowledgeMemoryBridge] = None
    
    @property
    def kb(self) -> KnowledgeForge:
        """Lazy-load knowledge forge."""
        if self._kb is None:
            self._kb = KnowledgeForge(DEFAULT_KB_PATH)
        return self._kb
    
    @property
    def bridge(self) -> Optional[KnowledgeMemoryBridge]:
        """Lazy-load bridge if memory_forge available."""
        if self._bridge is None:
            if ForgeDetector.is_available("memory_forge"):
                try:
                    self._bridge = KnowledgeMemoryBridge()
                except Exception:
                    pass
        return self._bridge
    
    @eidosian()
    def register_commands(self, subparsers) -> None:
        """Register knowledge-specific commands."""
        
        # List command
        list_parser = subparsers.add_parser(
            "list",
            help="List knowledge nodes",
        )
        list_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=20,
            help="Maximum items (default: 20)",
        )
        list_parser.add_argument(
            "-t", "--tag",
            help="Filter by tag",
        )
        list_parser.add_argument(
            "-c", "--concept",
            help="Filter by concept",
        )
        list_parser.set_defaults(func=self._cmd_list)
        
        # Search command
        search_parser = subparsers.add_parser(
            "search",
            help="Search knowledge graph",
        )
        search_parser.add_argument(
            "query",
            help="Search query",
        )
        search_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        search_parser.set_defaults(func=self._cmd_search)
        
        # Add command
        add_parser = subparsers.add_parser(
            "add",
            help="Add knowledge to graph",
        )
        add_parser.add_argument(
            "content",
            nargs="?",
            help="Content to add",
        )
        add_parser.add_argument(
            "-t", "--tags",
            nargs="+",
            help="Tags for the node",
        )
        add_parser.add_argument(
            "-c", "--concepts",
            nargs="+",
            help="Concepts to associate",
        )
        add_parser.add_argument(
            "-f", "--file",
            help="Read content from file",
        )
        add_parser.set_defaults(func=self._cmd_add)
        
        # Link command
        link_parser = subparsers.add_parser(
            "link",
            help="Link two knowledge nodes",
        )
        link_parser.add_argument(
            "node_a",
            help="First node ID (prefix OK)",
        )
        link_parser.add_argument(
            "node_b",
            help="Second node ID (prefix OK)",
        )
        link_parser.set_defaults(func=self._cmd_link)
        
        # Path command
        path_parser = subparsers.add_parser(
            "path",
            help="Find path between nodes",
        )
        path_parser.add_argument(
            "start",
            help="Start node ID (prefix OK)",
        )
        path_parser.add_argument(
            "end",
            help="End node ID (prefix OK)",
        )
        path_parser.set_defaults(func=self._cmd_path)
        
        # Concepts command
        concepts_parser = subparsers.add_parser(
            "concepts",
            help="List all concepts",
        )
        concepts_parser.set_defaults(func=self._cmd_concepts)
        
        # Unified search command
        unified_parser = subparsers.add_parser(
            "unified",
            help="Unified search across knowledge and memory",
        )
        unified_parser.add_argument(
            "query",
            help="Search query",
        )
        unified_parser.add_argument(
            "-n", "--limit",
            type=int,
            default=10,
            help="Maximum results (default: 10)",
        )
        unified_parser.set_defaults(func=self._cmd_unified)
        
        # Stats command
        stats_parser = subparsers.add_parser(
            "stats",
            help="Show detailed statistics",
        )
        stats_parser.set_defaults(func=self._cmd_stats)
        
        # Delete command
        delete_parser = subparsers.add_parser(
            "delete",
            help="Delete a knowledge node",
        )
        delete_parser.add_argument(
            "node_id",
            help="Node ID (prefix OK)",
        )
        delete_parser.add_argument(
            "-f", "--force",
            action="store_true",
            help="Skip confirmation",
        )
        delete_parser.set_defaults(func=self._cmd_delete)
    
    @eidosian()
    def cmd_status(self, args) -> CommandResult:
        """Show knowledge graph status."""
        try:
            stats = self.kb.stats()
            
            integrations = []
            if ForgeDetector.is_available("memory_forge"):
                integrations.append("memory_forge")
            if ForgeDetector.is_available("graphrag"):
                integrations.append("graphrag")
            
            return CommandResult(
                True,
                f"Knowledge graph operational - {stats['node_count']} nodes, {stats['concept_count']} concepts",
                {
                    "nodes": stats["node_count"],
                    "concepts": stats["concept_count"],
                    "kb_path": str(DEFAULT_KB_PATH),
                    "kb_exists": DEFAULT_KB_PATH.exists(),
                    "integrations": integrations,
                }
            )
        except Exception as e:
            return CommandResult(False, f"Error accessing knowledge graph: {e}")
    
    def _cmd_list(self, args) -> None:
        """List knowledge nodes."""
        try:
            if args.tag:
                nodes = self.kb.get_by_tag(args.tag)
            elif args.concept:
                nodes = self.kb.get_by_concept(args.concept)
            else:
                nodes = self.kb.list_nodes(args.limit)
            
            items = []
            for node in nodes[:args.limit]:
                content_preview = str(node.content)[:80]
                if len(str(node.content)) > 80:
                    content_preview += "..."
                items.append({
                    "id": node.id[:8],
                    "content": content_preview,
                    "tags": list(node.tags)[:3],
                    "links": len(node.links),
                })
            
            result = CommandResult(
                True,
                f"Found {len(items)} nodes",
                {"nodes": items, "total": len(items)}
            )
        except Exception as e:
            result = CommandResult(False, f"Error listing nodes: {e}")
        self._output(result, args)
    
    def _cmd_search(self, args) -> None:
        """Search knowledge graph."""
        try:
            nodes = self.kb.search(args.query)[:args.limit]
            
            results = []
            for node in nodes:
                content_preview = str(node.content)[:100]
                if len(str(node.content)) > 100:
                    content_preview += "..."
                results.append({
                    "id": node.id[:8],
                    "content": content_preview,
                    "tags": list(node.tags),
                })
            
            result = CommandResult(
                True,
                f"Found {len(results)} matching nodes",
                {"results": results, "count": len(results)}
            )
        except Exception as e:
            result = CommandResult(False, f"Search error: {e}")
        self._output(result, args)
    
    def _cmd_add(self, args) -> None:
        """Add knowledge to graph."""
        try:
            content = args.content
            if args.file:
                content = Path(args.file).read_text()
            
            if not content:
                result = CommandResult(False, "Content required")
            else:
                node = self.kb.add_knowledge(
                    content=content,
                    concepts=args.concepts,
                    tags=args.tags,
                )
                result = CommandResult(
                    True,
                    f"Added node {node.id[:8]}",
                    {"id": node.id, "tags": list(node.tags)}
                )
        except Exception as e:
            result = CommandResult(False, f"Error adding node: {e}")
        self._output(result, args)
    
    def _cmd_link(self, args) -> None:
        """Link two knowledge nodes."""
        try:
            node_a = self._find_node_by_prefix(args.node_a)
            node_b = self._find_node_by_prefix(args.node_b)
            
            if not node_a:
                result = CommandResult(False, f"Node not found: {args.node_a}")
            elif not node_b:
                result = CommandResult(False, f"Node not found: {args.node_b}")
            else:
                self.kb.link_nodes(node_a, node_b)
                result = CommandResult(
                    True,
                    f"Linked {node_a[:8]} ↔ {node_b[:8]}",
                    {"linked": [node_a[:8], node_b[:8]]}
                )
        except Exception as e:
            result = CommandResult(False, f"Error linking nodes: {e}")
        self._output(result, args)
    
    def _cmd_path(self, args) -> None:
        """Find path between nodes."""
        try:
            start = self._find_node_by_prefix(args.start)
            end = self._find_node_by_prefix(args.end)
            
            if not start:
                result = CommandResult(False, f"Node not found: {args.start}")
            elif not end:
                result = CommandResult(False, f"Node not found: {args.end}")
            else:
                path = self.kb.find_path(start, end)
                if not path:
                    result = CommandResult(True, "No path found", {"path": None})
                else:
                    result = CommandResult(
                        True,
                        f"Path: {' → '.join(p[:8] for p in path)}",
                        {"path": [p[:8] for p in path], "length": len(path)}
                    )
        except Exception as e:
            result = CommandResult(False, f"Error finding path: {e}")
        self._output(result, args)
    
    def _cmd_concepts(self, args) -> None:
        """List all concepts."""
        try:
            stats = self.kb.stats()
            concepts = stats["concepts"]
            
            concept_data = []
            for concept in concepts[:50]:
                nodes = self.kb.get_by_concept(concept)
                concept_data.append({
                    "concept": concept,
                    "node_count": len(nodes),
                })
            
            result = CommandResult(
                True,
                f"Found {len(concepts)} concepts",
                {"concepts": concept_data}
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _cmd_unified(self, args) -> None:
        """Unified search across knowledge and memory."""
        if not self.bridge:
            result = CommandResult(False, "Unified search requires memory_forge integration")
        else:
            try:
                results = self.bridge.unified_search(args.query, top_k=args.limit)
                
                items = []
                for r in results:
                    items.append({
                        "source": r.source,
                        "content": r.content[:80] + "..." if len(r.content) > 80 else r.content,
                        "score": round(r.score, 3) if r.score else None,
                    })
                
                result = CommandResult(
                    True,
                    f"Found {len(items)} results across knowledge and memory",
                    {"results": items}
                )
            except Exception as e:
                result = CommandResult(False, f"Search error: {e}")
        self._output(result, args)
    
    def _cmd_stats(self, args) -> None:
        """Show detailed statistics."""
        try:
            stats = self.kb.stats()
            
            total_links = sum(len(n.links) for n in self.kb.nodes.values())
            all_tags = set()
            for n in self.kb.nodes.values():
                all_tags.update(n.tags)
            
            data = {
                "nodes": stats["node_count"],
                "concepts": stats["concept_count"],
                "unique_tags": len(all_tags),
                "total_links": total_links,
                "avg_links_per_node": round(total_links / max(1, stats["node_count"]), 2),
                "top_concepts": stats["concepts"][:10],
                "top_tags": list(all_tags)[:10],
            }
            
            result = CommandResult(
                True,
                f"Graph: {data['nodes']} nodes, {data['concepts']} concepts, {data['total_links']} links",
                data
            )
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _cmd_delete(self, args) -> None:
        """Delete a knowledge node."""
        try:
            node_id = self._find_node_by_prefix(args.node_id)
            if not node_id:
                result = CommandResult(False, f"Node not found: {args.node_id}")
            else:
                success = self.kb.delete_node(node_id)
                if success:
                    result = CommandResult(
                        True,
                        f"Deleted node {node_id[:8]}",
                        {"deleted": node_id[:8]}
                    )
                else:
                    result = CommandResult(False, "Delete failed")
        except Exception as e:
            result = CommandResult(False, f"Error: {e}")
        self._output(result, args)
    
    def _find_node_by_prefix(self, prefix: str) -> Optional[str]:
        """Find full node ID from prefix."""
        for node_id in self.kb.nodes:
            if node_id.startswith(prefix):
                return node_id
        return None


@eidosian()
def main():
    """Entry point for knowledge-forge CLI."""
    cli = KnowledgeForgeCLI()
    sys.exit(cli.run())


if __name__ == "__main__":
    main()
