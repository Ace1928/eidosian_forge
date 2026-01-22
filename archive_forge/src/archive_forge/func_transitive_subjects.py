from rdflib.namespace import RDF
from rdflib.paths import Path
from rdflib.term import BNode, Node, URIRef
def transitive_subjects(self, predicate, remember=None):
    return self._resources(self._graph.transitive_subjects(predicate, self._identifier, remember))