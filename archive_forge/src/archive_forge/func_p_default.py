from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def p_default(self, node, position, newline=False):
    if position != SUBJECT and (not newline):
        self.write(' ')
    self.write(self.label(node, position))
    return True