import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def phase3(self):
    potential_edges = combinations(self.remaining_degree, 2)
    H = nx.Graph([(u, v) for u, v in potential_edges if not self.graph.has_edge(u, v)])
    rng = self.rng
    while self.remaining_degree:
        if not self.suitable_edge():
            raise nx.NetworkXUnfeasible('no suitable edges left')
        while True:
            u, v = sorted(rng.choice(list(H.edges())))
            if rng.random() < self.q(u, v):
                break
        if rng.random() < self.p(u, v):
            self.graph.add_edge(u, v)
            self.update_remaining(u, v, aux_graph=H)