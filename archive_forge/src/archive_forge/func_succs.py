import networkx as nx
from .isomorphvf2 import DiGraphMatcher, GraphMatcher
def succs(self, Gx, core_x, v, Gx_node=None):
    succ = [n for n in Gx.successors(v) if n in core_x]
    if Gx_node:
        succ.append(Gx_node)
    return succ