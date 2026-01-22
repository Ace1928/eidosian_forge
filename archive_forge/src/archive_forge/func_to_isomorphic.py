from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def to_isomorphic(graph: Graph) -> IsomorphicGraph:
    if isinstance(graph, IsomorphicGraph):
        return graph
    result = IsomorphicGraph()
    if hasattr(graph, 'identifier'):
        result = IsomorphicGraph(identifier=graph.identifier)
    result += graph
    return result