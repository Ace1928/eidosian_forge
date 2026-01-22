from contextlib import contextmanager
from rdflib.graph import Graph
from rdflib.namespace import RDF
from rdflib.term import BNode, Identifier, Literal, URIRef
def cast_value(v, **kws):
    if not isinstance(v, Literal):
        v = Literal(v, **kws)
    return v