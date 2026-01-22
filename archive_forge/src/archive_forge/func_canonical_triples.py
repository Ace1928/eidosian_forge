from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def canonical_triples(self, stats: Optional[Stats]=None):
    if stats is not None:
        start_coloring = datetime.now()
    coloring = self._initial_color()
    if stats is not None:
        stats['triple_count'] = len(self.graph)
        stats['adjacent_nodes'] = max(0, len(coloring) - 1)
    coloring = self._refine(coloring, coloring[:])
    if stats is not None:
        stats['initial_coloring_runtime'] = _total_seconds(datetime.now() - start_coloring)
        stats['initial_color_count'] = len(coloring)
    if not self._discrete(coloring):
        depth = [0]
        coloring = self._traces(coloring, stats=stats, depth=depth)
        if stats is not None:
            stats['tree_depth'] = depth[0]
    elif stats is not None:
        stats['individuations'] = 0
        stats['tree_depth'] = 0
    if stats is not None:
        stats['color_count'] = len(coloring)
    bnode_labels: Dict[Node, str] = dict([(c.nodes[0], c.hash_color()) for c in coloring])
    if stats is not None:
        stats['canonicalize_triples_runtime'] = _total_seconds(datetime.now() - start_coloring)
    for triple in self.graph:
        result = tuple(self._canonicalize_bnodes(triple, bnode_labels))
        yield result