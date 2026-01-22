import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def preds(self, Gx, core_x, v, Gx_node=None):
    pred = [n for n in Gx.predecessors(v) if n in core_x]
    if Gx_node:
        pred.append(Gx_node)
    return pred