from itertools import combinations
from rdflib import BNode, Graph
def internal_hash(self):
    """
        This is defined instead of __hash__ to avoid a circular recursion
        scenario with the Memory store for rdflib which requires a hash
        lookup in order to return a generator of triples
        """
    return hash(tuple(sorted(self.hashtriples())))