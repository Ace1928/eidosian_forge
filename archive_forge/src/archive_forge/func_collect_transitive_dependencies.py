from typing import Any, Dict, List, Set
from ..language import (
def collect_transitive_dependencies(collected: Set[str], dep_graph: DepGraph, from_name: str) -> None:
    """Collect transitive dependencies.

    From a dependency graph, collects a list of transitive dependencies by recursing
    through a dependency graph.
    """
    if from_name not in collected:
        collected.add(from_name)
        immediate_deps = dep_graph.get(from_name)
        if immediate_deps is not None:
            for to_name in immediate_deps:
                collect_transitive_dependencies(collected, dep_graph, to_name)