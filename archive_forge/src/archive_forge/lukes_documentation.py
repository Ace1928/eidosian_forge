from copy import deepcopy
from functools import lru_cache
from random import choice
import networkx as nx
from networkx.utils import not_implemented_for
Optimal partitioning of a weighted tree using the Lukes algorithm.

    This algorithm partitions a connected, acyclic graph featuring integer
    node weights and float edge weights. The resulting clusters are such
    that the total weight of the nodes in each cluster does not exceed
    max_size and that the weight of the edges that are cut by the partition
    is minimum. The algorithm is based on [1]_.

    Parameters
    ----------
    G : NetworkX graph

    max_size : int
        Maximum weight a partition can have in terms of sum of
        node_weight for all nodes in the partition

    edge_weight : key
        Edge data key to use as weight. If None, the weights are all
        set to one.

    node_weight : key
        Node data key to use as weight. If None, the weights are all
        set to one. The data must be int.

    Returns
    -------
    partition : list
        A list of sets of nodes representing the clusters of the
        partition.

    Raises
    ------
    NotATree
        If G is not a tree.
    TypeError
        If any of the values of node_weight is not int.

    References
    ----------
    .. [1] Lukes, J. A. (1974).
       "Efficient Algorithm for the Partitioning of Trees."
       IBM Journal of Research and Development, 18(3), 217–224.

    