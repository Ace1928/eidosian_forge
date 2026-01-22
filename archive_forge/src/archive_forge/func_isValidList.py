from rdflib.exceptions import Error
from rdflib.namespace import RDF
from rdflib.term import BNode, Literal, URIRef
from .turtle import RecursiveSerializer
def isValidList(self, l_):
    """
        Checks if l is a valid RDF list, i.e. no nodes have other properties.
        """
    try:
        if self.store.value(l_, RDF.first) is None:
            return False
    except Exception:
        return False
    while l_:
        if l_ != RDF.nil and len(list(self.store.predicate_objects(l_))) != 2:
            return False
        l_ = self.store.value(l_, RDF.rest)
    return True