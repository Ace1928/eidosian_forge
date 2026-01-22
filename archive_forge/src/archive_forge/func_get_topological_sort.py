from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def get_topological_sort(self) -> List[str]:
    """Get a list of entity names in the graph sorted by causal dependence."""
    import networkx as nx
    return list(nx.topological_sort(self._graph))