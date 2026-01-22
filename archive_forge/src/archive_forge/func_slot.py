import networkx as nx
from collections import deque
def slot(self, vertex):
    try:
        return self.slots[self.index(vertex)]
    except ValueError:
        raise ValueError('Vertex is not an end of this edge.')