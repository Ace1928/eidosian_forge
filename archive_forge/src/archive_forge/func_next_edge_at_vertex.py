import networkx
import random
from .links import Strand
from ..graphs import CyclicList, Digraph
from collections import namedtuple, Counter
def next_edge_at_vertex(self, edge, vertex):
    """
        The next edge at the vertex
        """
    link = self.link(vertex)
    return link[(link.index(edge) + 1) % len(link)]