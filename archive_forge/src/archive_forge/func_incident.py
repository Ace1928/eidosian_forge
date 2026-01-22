import networkx as nx
from collections import deque
def incident(self, vertex):
    """
        Return the set of non-loops incident to the vertex.
        """
    return set((e for e in self.incidence_dict[vertex] if not e.is_loop()))