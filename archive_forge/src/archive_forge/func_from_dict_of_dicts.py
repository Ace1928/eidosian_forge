import warnings
from collections.abc import Collection, Generator, Iterator
import networkx as nx
@nx._dispatch(graphs=None)
def from_dict_of_dicts(d, create_using=None, multigraph_input=False):
    """Returns a graph from a dictionary of dictionaries.

    Parameters
    ----------
    d : dictionary of dictionaries
      A dictionary of dictionaries adjacency representation.

    create_using : NetworkX graph constructor, optional (default=nx.Graph)
        Graph type to create. If graph instance, then cleared before populated.

    multigraph_input : bool (default False)
       When True, the dict `d` is assumed
       to be a dict-of-dict-of-dict-of-dict structure keyed by
       node to neighbor to edge keys to edge data for multi-edges.
       Otherwise this routine assumes dict-of-dict-of-dict keyed by
       node to neighbor to edge data.

    Examples
    --------
    >>> dod = {0: {1: {"weight": 1}}}  # single edge (0,1)
    >>> G = nx.from_dict_of_dicts(dod)

    or

    >>> G = nx.Graph(dod)  # use Graph constructor

    """
    G = nx.empty_graph(0, create_using)
    G.add_nodes_from(d)
    if multigraph_input:
        if G.is_directed():
            if G.is_multigraph():
                G.add_edges_from(((u, v, key, data) for u, nbrs in d.items() for v, datadict in nbrs.items() for key, data in datadict.items()))
            else:
                G.add_edges_from(((u, v, data) for u, nbrs in d.items() for v, datadict in nbrs.items() for key, data in datadict.items()))
        elif G.is_multigraph():
            seen = set()
            for u, nbrs in d.items():
                for v, datadict in nbrs.items():
                    if (u, v) not in seen:
                        G.add_edges_from(((u, v, key, data) for key, data in datadict.items()))
                        seen.add((v, u))
        else:
            seen = set()
            for u, nbrs in d.items():
                for v, datadict in nbrs.items():
                    if (u, v) not in seen:
                        G.add_edges_from(((u, v, data) for key, data in datadict.items()))
                        seen.add((v, u))
    elif G.is_multigraph() and (not G.is_directed()):
        seen = set()
        for u, nbrs in d.items():
            for v, data in nbrs.items():
                if (u, v) not in seen:
                    G.add_edge(u, v, key=0)
                    G[u][v][0].update(data)
                seen.add((v, u))
    else:
        G.add_edges_from(((u, v, data) for u, nbrs in d.items() for v, data in nbrs.items()))
    return G