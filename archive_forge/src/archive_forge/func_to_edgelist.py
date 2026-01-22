import warnings
from collections.abc import Collection, Generator, Iterator
import networkx as nx
@nx._dispatch(preserve_edge_attrs=True)
def to_edgelist(G, nodelist=None):
    """Returns a list of edges in the graph.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list
       Use only nodes specified in nodelist

    """
    if nodelist is None:
        return G.edges(data=True)
    return G.edges(nodelist, data=True)