from __future__ import annotations
import logging
import pathlib
import random
from io import BytesIO
from typing import (
from urllib.parse import urlparse
from urllib.request import url2pathname
import rdflib.exceptions as exceptions
import rdflib.namespace as namespace  # noqa: F401 # This is here because it is used in a docstring.
import rdflib.plugin as plugin
import rdflib.query as query
import rdflib.util  # avoid circular dependency
from rdflib.collection import Collection
from rdflib.exceptions import ParserError
from rdflib.namespace import RDF, Namespace, NamespaceManager
from rdflib.parser import InputSource, Parser, create_input_source
from rdflib.paths import Path
from rdflib.resource import Resource
from rdflib.serializer import Serializer
from rdflib.store import Store
from rdflib.term import (
def quads(self, triple_or_quad: _TripleOrQuadSelectorType) -> Generator[Tuple['_SubjectType', Union[Path, '_PredicateType'], '_ObjectType', '_ContextType'], None, None]:
    """Iterate over all the quads in the entire aggregate graph"""
    c = None
    if len(triple_or_quad) == 4:
        s, p, o, c = triple_or_quad
    else:
        s, p, o = triple_or_quad
    if c is not None:
        for graph in [g for g in self.graphs if g == c]:
            for s1, p1, o1 in graph.triples((s, p, o)):
                yield (s1, p1, o1, graph)
    else:
        for graph in self.graphs:
            for s1, p1, o1 in graph.triples((s, p, o)):
                yield (s1, p1, o1, graph)