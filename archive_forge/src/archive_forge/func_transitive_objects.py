from rdflib.namespace import RDF
from rdflib.paths import Path
from rdflib.term import BNode, Node, URIRef
def transitive_objects(self, predicate, remember=None):
    return self._resources(self._graph.transitive_objects(self._identifier, predicate, remember))