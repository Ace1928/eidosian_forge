import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def basic_topological_numbering(G):
    """
    Finds an optimal weighted topological numbering a directed acyclic graph
    """
    in_valences = dict(((v, G.indegree(v)) for v in G.vertices))
    numbering = {}
    curr_sources = [v for v, i in in_valences.items() if i == 0]
    curr_number = 0
    while len(in_valences):
        new_sources = []
        for v in curr_sources:
            in_valences.pop(v)
            numbering[v] = curr_number
            for e in G.outgoing(v):
                w = e.head
                in_valences[w] -= 1
                if in_valences[w] == 0:
                    new_sources.append(w)
            curr_sources = new_sources
        curr_number += 1
    return numbering