import networkx as nx
from collections import deque
def multi_valence(self, vertex):
    """
        Return the valence of a vertex, counting edge multiplicities.
        """
    return sum((e.multiplicity for e in self.incidence_dict[vertex]))