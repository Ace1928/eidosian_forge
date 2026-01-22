from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for, pairwise
def path_weight(G, path, weight):
    """Returns total cost associated with specified path and weight

    Parameters
    ----------
    G : graph
        A NetworkX graph.

    path: list
        A list of node labels which defines the path to traverse

    weight: string
        A string indicating which edge attribute to use for path cost

    Returns
    -------
    cost: int or float
        An integer or a float representing the total cost with respect to the
        specified weight of the specified path

    Raises
    ------
    NetworkXNoPath
        If the specified edge does not exist.
    """
    multigraph = G.is_multigraph()
    cost = 0
    if not nx.is_path(G, path):
        raise nx.NetworkXNoPath('path does not exist')
    for node, nbr in nx.utils.pairwise(path):
        if multigraph:
            cost += min((v[weight] for v in G[node][nbr].values()))
        else:
            cost += G[node][nbr][weight]
    return cost