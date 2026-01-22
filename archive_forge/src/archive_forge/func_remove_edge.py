from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def remove_edge(self, s, t):
    """
        Remove an edge (s, t) where parent[t] == s from the spanning tree.
        """
    size_t = self.subtree_size[t]
    prev_t = self.prev_node_dft[t]
    last_t = self.last_descendent_dft[t]
    next_last_t = self.next_node_dft[last_t]
    self.parent[t] = None
    self.parent_edge[t] = None
    self.next_node_dft[prev_t] = next_last_t
    self.prev_node_dft[next_last_t] = prev_t
    self.next_node_dft[last_t] = t
    self.prev_node_dft[t] = last_t
    while s is not None:
        self.subtree_size[s] -= size_t
        if self.last_descendent_dft[s] == last_t:
            self.last_descendent_dft[s] = prev_t
        s = self.parent[s]