from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def s_squared(self, subject):
    if self._references[subject] > 0 or not isinstance(subject, BNode):
        return False
    self.write('\n' + self.indent() + '[]')
    self.predicateList(subject, newline=False)
    self.write(' ;\n.')
    return True