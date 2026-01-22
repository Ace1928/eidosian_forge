import networkx as nx
from collections import deque
def outdegree(self, vertex):
    return len([e for e in self.incidence_dict[vertex] if e.tail is vertex])