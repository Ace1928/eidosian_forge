from copy import deepcopy
import networkx as nx
from networkx.algorithms.centrality.betweenness import (
from networkx.utils.decorators import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def prominent_group(G, k, weight=None, C=None, endpoints=False, normalized=True, greedy=False):
    """Find the prominent group of size $k$ in graph $G$. The prominence of the
    group is evaluated by the group betweenness centrality.

    Group betweenness centrality of a group of nodes $C$ is the sum of the
    fraction of all-pairs shortest paths that pass through any vertex in $C$

    .. math::

       c_B(v) =\\sum_{s,t \\in V} \\frac{\\sigma(s, t|v)}{\\sigma(s, t)}

    where $V$ is the set of nodes, $\\sigma(s, t)$ is the number of
    shortest $(s, t)$-paths, and $\\sigma(s, t|C)$ is the number of
    those paths passing through some node in group $C$. Note that
    $(s, t)$ are not members of the group ($V-C$ is the set of nodes
    in $V$ that are not in $C$).

    Parameters
    ----------
    G : graph
       A NetworkX graph.

    k : int
       The number of nodes in the group.

    normalized : bool, optional (default=True)
       If True, group betweenness is normalized by ``1/((|V|-|C|)(|V|-|C|-1))``
       where ``|V|`` is the number of nodes in G and ``|C|`` is the number of
       nodes in C.

    weight : None or string, optional (default=None)
       If None, all edge weights are considered equal.
       Otherwise holds the name of the edge attribute used as weight.
       The weight of an edge is treated as the length or distance between the two sides.

    endpoints : bool, optional (default=False)
       If True include the endpoints in the shortest path counts.

    C : list or set, optional (default=None)
       list of nodes which won't be candidates of the prominent group.

    greedy : bool, optional (default=False)
       Using a naive greedy algorithm in order to find non-optimal prominent
       group. For scale free networks the results are negligibly below the optimal
       results.

    Raises
    ------
    NodeNotFound
       If node(s) in C are not present in G.

    Returns
    -------
    max_GBC : float
       The group betweenness centrality of the prominent group.

    max_group : list
        The list of nodes in the prominent group.

    See Also
    --------
    betweenness_centrality, group_betweenness_centrality

    Notes
    -----
    Group betweenness centrality is described in [1]_ and its importance discussed in [3]_.
    The algorithm is described in [2]_ and is based on techniques mentioned in [4]_.

    The number of nodes in the group must be a maximum of ``n - 2`` where ``n``
    is the total number of nodes in the graph.

    For weighted graphs the edge weights must be greater than zero.
    Zero edge weights can produce an infinite number of equal length
    paths between pairs of nodes.

    The total number of paths between source and target is counted
    differently for directed and undirected graphs. Directed paths
    between "u" and "v" are counted as two possible paths (one each
    direction) while undirected paths between "u" and "v" are counted
    as one path. Said another way, the sum in the expression above is
    over all ``s != t`` for directed graphs and for ``s < t`` for undirected graphs.

    References
    ----------
    .. [1] M G Everett and S P Borgatti:
       The Centrality of Groups and Classes.
       Journal of Mathematical Sociology. 23(3): 181-201. 1999.
       http://www.analytictech.com/borgatti/group_centrality.htm
    .. [2] Rami Puzis, Yuval Elovici, and Shlomi Dolev:
       "Finding the Most Prominent Group in Complex Networks"
       AI communications 20(4): 287-296, 2007.
       https://www.researchgate.net/profile/Rami_Puzis2/publication/220308855
    .. [3] Sourav Medya et. al.:
       Group Centrality Maximization via Network Design.
       SIAM International Conference on Data Mining, SDM 2018, 126–134.
       https://sites.cs.ucsb.edu/~arlei/pubs/sdm18.pdf
    .. [4] Rami Puzis, Yuval Elovici, and Shlomi Dolev.
       "Fast algorithm for successive computation of group betweenness centrality."
       https://journals.aps.org/pre/pdf/10.1103/PhysRevE.76.056709
    """
    import numpy as np
    import pandas as pd
    if C is not None:
        C = set(C)
        if C - G.nodes:
            raise nx.NodeNotFound(f'The node(s) {C - G.nodes} are in C but not in G.')
        nodes = list(G.nodes - C)
    else:
        nodes = list(G.nodes)
    DF_tree = nx.Graph()
    PB, sigma, D = _group_preprocessing(G, nodes, weight)
    betweenness = pd.DataFrame.from_dict(PB)
    if C is not None:
        for node in C:
            betweenness.drop(index=node, inplace=True)
            betweenness.drop(columns=node, inplace=True)
    CL = [node for _, node in sorted(zip(np.diag(betweenness), nodes), reverse=True)]
    max_GBC = 0
    max_group = []
    DF_tree.add_node(1, CL=CL, betweenness=betweenness, GBC=0, GM=[], sigma=sigma, cont=dict(zip(nodes, np.diag(betweenness))))
    DF_tree.nodes[1]['heu'] = 0
    for i in range(k):
        DF_tree.nodes[1]['heu'] += DF_tree.nodes[1]['cont'][DF_tree.nodes[1]['CL'][i]]
    max_GBC, DF_tree, max_group = _dfbnb(G, k, DF_tree, max_GBC, 1, D, max_group, nodes, greedy)
    v = len(G)
    if not endpoints:
        scale = 0
        if nx.is_directed(G):
            if nx.is_strongly_connected(G):
                scale = k * (2 * v - k - 1)
        elif nx.is_connected(G):
            scale = k * (2 * v - k - 1)
        if scale == 0:
            for group_node1 in max_group:
                for node in D[group_node1]:
                    if node != group_node1:
                        if node in max_group:
                            scale += 1
                        else:
                            scale += 2
        max_GBC -= scale
    if normalized:
        scale = 1 / ((v - k) * (v - k - 1))
        max_GBC *= scale
    elif not G.is_directed():
        max_GBC /= 2
    max_GBC = float('%.2f' % max_GBC)
    return (max_GBC, max_group)