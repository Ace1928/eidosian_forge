import heapq
import math
from itertools import chain, combinations, zip_longest
from operator import itemgetter
import networkx as nx
from networkx.utils import py_random_state, random_weighted_sample
def phase2(self):
    remaining_deg = self.remaining_degree
    rng = self.rng
    while len(remaining_deg) >= 2 * self.dmax:
        while True:
            u, v = sorted(rng.sample(list(remaining_deg.keys()), 2))
            if self.graph.has_edge(u, v):
                continue
            if rng.random() < self.q(u, v):
                break
        if rng.random() < self.p(u, v):
            self.graph.add_edge(u, v)
            self.update_remaining(u, v)