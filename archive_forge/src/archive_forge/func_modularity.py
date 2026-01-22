from itertools import combinations
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils.decorators import argmap
@nx._dispatch(edge_attrs='weight')
def modularity(G, communities, weight='weight', resolution=1):
    """Returns the modularity of the given partition of the graph.

    Modularity is defined in [1]_ as

    .. math::
        Q = \\frac{1}{2m} \\sum_{ij} \\left( A_{ij} - \\gamma\\frac{k_ik_j}{2m}\\right)
            \\delta(c_i,c_j)

    where $m$ is the number of edges (or sum of all edge weights as in [5]_),
    $A$ is the adjacency matrix of `G`, $k_i$ is the (weighted) degree of $i$,
    $\\gamma$ is the resolution parameter, and $\\delta(c_i, c_j)$ is 1 if $i$ and
    $j$ are in the same community else 0.

    According to [2]_ (and verified by some algebra) this can be reduced to

    .. math::
       Q = \\sum_{c=1}^{n}
       \\left[ \\frac{L_c}{m} - \\gamma\\left( \\frac{k_c}{2m} \\right) ^2 \\right]

    where the sum iterates over all communities $c$, $m$ is the number of edges,
    $L_c$ is the number of intra-community links for community $c$,
    $k_c$ is the sum of degrees of the nodes in community $c$,
    and $\\gamma$ is the resolution parameter.

    The resolution parameter sets an arbitrary tradeoff between intra-group
    edges and inter-group edges. More complex grouping patterns can be
    discovered by analyzing the same network with multiple values of gamma
    and then combining the results [3]_. That said, it is very common to
    simply use gamma=1. More on the choice of gamma is in [4]_.

    The second formula is the one actually used in calculation of the modularity.
    For directed graphs the second formula replaces $k_c$ with $k^{in}_c k^{out}_c$.

    Parameters
    ----------
    G : NetworkX Graph

    communities : list or iterable of set of nodes
        These node sets must represent a partition of G's nodes.

    weight : string or None, optional (default="weight")
        The edge attribute that holds the numerical value used
        as a weight. If None or an edge does not have that attribute,
        then that edge has weight 1.

    resolution : float (default=1)
        If resolution is less than 1, modularity favors larger communities.
        Greater than 1 favors smaller communities.

    Returns
    -------
    Q : float
        The modularity of the partition.

    Raises
    ------
    NotAPartition
        If `communities` is not a partition of the nodes of `G`.

    Examples
    --------
    >>> G = nx.barbell_graph(3, 0)
    >>> nx.community.modularity(G, [{0, 1, 2}, {3, 4, 5}])
    0.35714285714285715
    >>> nx.community.modularity(G, nx.community.label_propagation_communities(G))
    0.35714285714285715

    References
    ----------
    .. [1] M. E. J. Newman "Networks: An Introduction", page 224.
       Oxford University Press, 2011.
    .. [2] Clauset, Aaron, Mark EJ Newman, and Cristopher Moore.
       "Finding community structure in very large networks."
       Phys. Rev. E 70.6 (2004). <https://arxiv.org/abs/cond-mat/0408187>
    .. [3] Reichardt and Bornholdt "Statistical Mechanics of Community Detection"
       Phys. Rev. E 74, 016110, 2006. https://doi.org/10.1103/PhysRevE.74.016110
    .. [4] M. E. J. Newman, "Equivalence between modularity optimization and
       maximum likelihood methods for community detection"
       Phys. Rev. E 94, 052315, 2016. https://doi.org/10.1103/PhysRevE.94.052315
    .. [5] Blondel, V.D. et al. "Fast unfolding of communities in large
       networks" J. Stat. Mech 10008, 1-12 (2008).
       https://doi.org/10.1088/1742-5468/2008/10/P10008
    """
    if not isinstance(communities, list):
        communities = list(communities)
    if not is_partition(G, communities):
        raise NotAPartition(G, communities)
    directed = G.is_directed()
    if directed:
        out_degree = dict(G.out_degree(weight=weight))
        in_degree = dict(G.in_degree(weight=weight))
        m = sum(out_degree.values())
        norm = 1 / m ** 2
    else:
        out_degree = in_degree = dict(G.degree(weight=weight))
        deg_sum = sum(out_degree.values())
        m = deg_sum / 2
        norm = 1 / deg_sum ** 2

    def community_contribution(community):
        comm = set(community)
        L_c = sum((wt for u, v, wt in G.edges(comm, data=weight, default=1) if v in comm))
        out_degree_sum = sum((out_degree[u] for u in comm))
        in_degree_sum = sum((in_degree[u] for u in comm)) if directed else out_degree_sum
        return L_c / m - resolution * out_degree_sum * in_degree_sum * norm
    return sum(map(community_contribution, communities))