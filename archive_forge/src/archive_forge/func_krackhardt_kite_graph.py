from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def krackhardt_kite_graph(create_using=None):
    """
    Returns the Krackhardt Kite Social Network.

    A 10 actor social network introduced by David Krackhardt
    to illustrate different centrality measures [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Krackhardt Kite graph with 10 nodes and 18 edges

    Notes
    -----
    The traditional labeling is:
    Andre=1, Beverley=2, Carol=3, Diane=4,
    Ed=5, Fernando=6, Garth=7, Heather=8, Ike=9, Jane=10.

    References
    ----------
    .. [1] Krackhardt, David. "Assessing the Political Landscape: Structure,
       Cognition, and Power in Organizations". Administrative Science Quarterly.
       35 (2): 342–369. doi:10.2307/2393394. JSTOR 2393394. June 1990.

    """
    G = nx.from_dict_of_lists({0: [1, 2, 3, 5], 1: [0, 3, 4, 6], 2: [0, 3, 5], 3: [0, 1, 2, 4, 5, 6], 4: [1, 3, 6], 5: [0, 2, 3, 6, 7], 6: [1, 3, 4, 5, 7], 7: [5, 6, 8], 8: [7, 9], 9: [8]}, create_using=create_using)
    G.name = 'Krackhardt Kite Social Network'
    return G