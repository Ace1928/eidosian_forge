import itertools
from collections import defaultdict
from collections.abc import Mapping
from functools import cached_property
import networkx as nx
from networkx.algorithms.approximation import local_node_connectivity
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for
def subgraph(self, nodes):
    """This subgraph method returns a full AntiGraph. Not a View"""
    nodes = set(nodes)
    G = _AntiGraph()
    G.add_nodes_from(nodes)
    for n in G:
        Gnbrs = G.adjlist_inner_dict_factory()
        G._adj[n] = Gnbrs
        for nbr, d in self._adj[n].items():
            if nbr in G._adj:
                Gnbrs[nbr] = d
                G._adj[nbr][n] = d
    G.graph = self.graph
    return G