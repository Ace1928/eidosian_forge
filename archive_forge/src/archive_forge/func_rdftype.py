from contextlib import contextmanager
from rdflib.graph import Graph
from rdflib.namespace import RDF
from rdflib.term import BNode, Identifier, Literal, URIRef
def rdftype(self, t):
    """
        Shorthand for setting rdf:type of the current subject.

        Usage::

            >>> from rdflib import URIRef
            >>> from rdflib.namespace import RDF, RDFS
            >>> d = Describer(about="http://example.org/")
            >>> d.rdftype(RDFS.Resource)
            >>> (URIRef('http://example.org/'),
            ...     RDF.type, RDFS.Resource) in d.graph
            True

        """
    self.graph.add((self._current(), RDF.type, t))