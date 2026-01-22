import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def chain_coordinates(self, kind):
    D = self.DAG_from_direction(kind)
    chain_coors = topological_numbering(D)
    return dict(((v, chain_coors[D.vertex_to_chain[v]]) for v in self.vertices))