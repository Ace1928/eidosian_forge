import random
import time
import networkx as nx
from networkx.algorithms.isomorphism.tree_isomorphism import (
from networkx.classes.function import is_directed
def check_isomorphism(t1, t2, isomorphism):
    mapping = {v2: v1 for v1, v2 in isomorphism}
    d1 = is_directed(t1)
    d2 = is_directed(t2)
    assert d1 == d2
    edges_1 = []
    for u, v in t1.edges():
        if d1:
            edges_1.append((u, v))
        elif u < v:
            edges_1.append((u, v))
        else:
            edges_1.append((v, u))
    edges_2 = []
    for u, v in t2.edges():
        u = mapping[u]
        v = mapping[v]
        if d2:
            edges_2.append((u, v))
        elif u < v:
            edges_2.append((u, v))
        else:
            edges_2.append((v, u))
    return sorted(edges_1) == sorted(edges_2)