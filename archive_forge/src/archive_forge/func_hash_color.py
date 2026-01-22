from __future__ import absolute_import, division, print_function
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import (
from rdflib.graph import ConjunctiveGraph, Graph, ReadOnlyGraphAggregate, _TripleType
from rdflib.term import BNode, IdentifiedNode, Node, URIRef
def hash_color(self, color: Optional[Tuple[ColorItem, ...]]=None) -> str:
    if color is None:
        color = self.color
    if color in self._hash_cache:
        return self._hash_cache[color]

    def stringify(x):
        if isinstance(x, Node):
            return x.n3()
        else:
            return str(x)
    if isinstance(color, Node):
        return stringify(color)
    value = 0
    for triple in color:
        value += self.hashfunc(' '.join([stringify(x) for x in triple]))
    val: str = '%x' % value
    self._hash_cache[color] = val
    return val