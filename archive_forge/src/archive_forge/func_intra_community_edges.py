from itertools import combinations
import networkx as nx
from networkx import NetworkXError
from networkx.algorithms.community.community_utils import is_partition
from networkx.utils.decorators import argmap
@nx._dispatch
def intra_community_edges(G, partition):
    """Returns the number of intra-community edges for a partition of `G`.

    Parameters
    ----------
    G : NetworkX graph.

    partition : iterable of sets of nodes
        This must be a partition of the nodes of `G`.

    The "intra-community edges" are those edges joining a pair of nodes
    in the same block of the partition.

    """
    return sum((G.subgraph(block).size() for block in partition))