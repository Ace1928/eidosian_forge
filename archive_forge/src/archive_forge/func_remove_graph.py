import collections
import re
from typing import (
from rdflib.graph import DATASET_DEFAULT_GRAPH_ID, Graph
from rdflib.plugins.stores.regexmatching import NATIVE_REGEX
from rdflib.store import Store
from rdflib.term import BNode, Identifier, Node, URIRef, Variable
from .sparqlconnector import SPARQLConnector
def remove_graph(self, graph: 'Graph') -> None:
    if not self.graph_aware:
        Store.remove_graph(self, graph)
    elif graph.identifier == DATASET_DEFAULT_GRAPH_ID:
        self.update('DROP DEFAULT')
    else:
        self.update('DROP GRAPH %s' % self.node_to_sparql(graph.identifier))