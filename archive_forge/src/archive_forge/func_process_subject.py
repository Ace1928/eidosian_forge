import warnings
from typing import IO, Optional
from rdflib.graph import Graph
from rdflib.namespace import RDF, XSD
from rdflib.serializer import Serializer
from rdflib.term import BNode, Literal, URIRef
from ..shared.jsonld.context import UNDEF, Context
from ..shared.jsonld.keys import CONTEXT, GRAPH, ID, LANG, LIST, SET, VOCAB
from ..shared.jsonld.util import json
def process_subject(self, graph, s, nodemap):
    if isinstance(s, URIRef):
        node_id = self.context.shrink_iri(s)
    elif isinstance(s, BNode):
        node_id = s.n3()
    else:
        node_id = None
    if node_id in nodemap:
        return None
    node = {}
    node[self.context.id_key] = node_id
    nodemap[node_id] = node
    for p, o in graph.predicate_objects(s):
        self.add_to_node(graph, s, p, o, node, nodemap)
    return node