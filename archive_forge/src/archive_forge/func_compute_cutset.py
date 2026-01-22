import pytest
import networkx as nx
from networkx.algorithms.flow import (
def compute_cutset(G, partition):
    reachable, non_reachable = partition
    cutset = set()
    for u, nbrs in ((n, G[n]) for n in reachable):
        cutset.update(((u, v) for v in nbrs if v in non_reachable))
    return cutset