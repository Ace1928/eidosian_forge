from itertools import chain, islice, repeat
from math import ceil, sqrt
import networkx as nx
from networkx.utils import not_implemented_for
def augment_flow(self, Wn, We, f):
    """
        Augment f units of flow along a cycle represented by Wn and We.
        """
    for i, p in zip(We, Wn):
        if self.edge_sources[i] == p:
            self.edge_flow[i] += f
        else:
            self.edge_flow[i] -= f