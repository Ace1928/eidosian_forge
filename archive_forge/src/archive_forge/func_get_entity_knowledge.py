from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def get_entity_knowledge(self, entity: str, depth: int=1) -> List[str]:
    """Get information about an entity."""
    import networkx as nx
    if not self._graph.has_node(entity):
        return []
    results = []
    for src, sink in nx.dfs_edges(self._graph, entity, depth_limit=depth):
        relation = self._graph[src][sink]['relation']
        results.append(f'{src} {relation} {sink}')
    return results