import pytest
import networkx as nx
@staticmethod
def pick_remove_edge(g):
    u = nx.utils.arbitrary_element(g)
    possible_nodes = list(g.neighbors(u))
    v = nx.utils.arbitrary_element(possible_nodes)
    return (u, v)