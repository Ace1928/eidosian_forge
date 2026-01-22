from __future__ import annotations
from typing import Any, List, NamedTuple, Optional, Tuple
def delete_triple(self, knowledge_triple: KnowledgeTriple) -> None:
    """Delete a triple from the graph."""
    if self._graph.has_edge(knowledge_triple.subject, knowledge_triple.object_):
        self._graph.remove_edge(knowledge_triple.subject, knowledge_triple.object_)