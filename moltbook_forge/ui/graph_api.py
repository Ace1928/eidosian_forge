#!/usr/bin/env python3
"""
Nexus Social Graph API.
Maps agent-to-agent relational links.
"""

from __future__ import annotations

from typing import Dict, List, Set
from pydantic import BaseModel

class GraphNode(BaseModel):
    id: str
    label: str
    type: str = "agent"
    val: int = 1

class GraphLink(BaseModel):
    source: str
    target: str
    type: str = "following"

class NexusGraph(BaseModel):
    nodes: List[GraphNode]
    links: List[GraphLink]

class SocialGraph:
    """Orchestrates relational mapping for the Moltiverse."""

    def __init__(self):
        self.nodes: Dict[str, GraphNode] = {}
        self.links: List[GraphLink] = []
        self.seen_links: Set[tuple] = set()

    def add_link(self, source: str, target: str):
        if (source, target) not in self.seen_links:
            if source not in self.nodes:
                self.nodes[source] = GraphNode(id=source, label=source)
            if target not in self.nodes:
                self.nodes[target] = GraphNode(id=target, label=target)
            
            self.links.append(GraphLink(source=source, target=target))
            self.seen_links.add((source, target))
            # Increase node weight based on incoming links
            self.nodes[target].val += 1

    def get_graph(self) -> NexusGraph:
        return NexusGraph(
            nodes=list(self.nodes.values()),
            links=self.links
        )
