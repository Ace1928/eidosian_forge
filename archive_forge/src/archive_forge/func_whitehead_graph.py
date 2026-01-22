from .graphs import ReducedGraph, Digraph, Poset
from collections import deque
import operator
def whitehead_graph(self):
    Wh = ReducedGraph()
    vertex_dict = {}
    for letter in self.generators:
        vertex_dict[letter] = letter
        vertex_dict[-letter] = -letter
        Wh.add_vertex(vertex_dict[letter])
        Wh.add_vertex(vertex_dict[-letter])
    for relator in self.relators:
        for n in range(-1, len(relator) - 1):
            Wh.add_edge(vertex_dict[relator[n]], vertex_dict[-relator[n + 1]])
    return Wh