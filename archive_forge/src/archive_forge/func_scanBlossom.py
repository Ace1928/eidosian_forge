from collections import Counter
from itertools import combinations, repeat
import networkx as nx
from networkx.utils import not_implemented_for
def scanBlossom(v, w):
    path = []
    base = NoNode
    while v is not NoNode:
        b = inblossom[v]
        if label[b] & 4:
            base = blossombase[b]
            break
        assert label[b] == 1
        path.append(b)
        label[b] = 5
        if labeledge[b] is None:
            assert blossombase[b] not in mate
            v = NoNode
        else:
            assert labeledge[b][0] == mate[blossombase[b]]
            v = labeledge[b][0]
            b = inblossom[v]
            assert label[b] == 2
            v = labeledge[b][0]
        if w is not NoNode:
            v, w = (w, v)
    for b in path:
        label[b] = 1
    return base