import networkx as nx
from collections import deque
def filled_euler(self):
    return len(self.vertices) - len(self.edges) + len(self.boundary_cycles())