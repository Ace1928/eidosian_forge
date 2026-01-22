from __future__ import annotations
import pickle
from io import BytesIO
from typing import (
from rdflib.events import Dispatcher, Event
@property
def node_pickler(self) -> NodePickler:
    if self.__node_pickler is None:
        from rdflib.graph import Graph, QuotedGraph
        from rdflib.term import BNode, Literal, URIRef, Variable
        self.__node_pickler = np = NodePickler()
        np.register(self, 'S')
        np.register(URIRef, 'U')
        np.register(BNode, 'B')
        np.register(Literal, 'L')
        np.register(Graph, 'G')
        np.register(QuotedGraph, 'Q')
        np.register(Variable, 'V')
    return self.__node_pickler