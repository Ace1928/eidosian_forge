from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def get_triples(self) -> List[Tuple[str, str, str]]:
    """Get all triples in the graph."""
    return [(u, v, d['relation']) for u, v, d in self._graph.edges(data=True)]