from itertools import combinations
from rdflib import BNode, Graph
def vhash(self, term, done=False):
    return tuple(sorted(self.vhashtriples(term, done)))