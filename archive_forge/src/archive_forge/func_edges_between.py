import networkx as nx
from collections import deque
def edges_between(self, vertex1, vertex2):
    return self.incident(vertex1).intersection(self.incident(vertex2))