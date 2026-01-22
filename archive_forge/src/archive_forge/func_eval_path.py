from __future__ import annotations
import warnings
from functools import total_ordering
from typing import (
from rdflib.term import Node, URIRef
def eval_path(graph: Graph, t: Tuple[Optional['_SubjectType'], Union[None, Path, _PredicateType], Optional['_ObjectType']]) -> Iterator[Tuple[_SubjectType, _ObjectType]]:
    return ((s, o) for s, p, o in graph.triples(t))