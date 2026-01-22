from pyomo.common.deprecation import deprecated
from pyomo.contrib.incidence_analysis.matching import maximum_matching
from pyomo.contrib.incidence_analysis.common.dulmage_mendelsohn import (
from pyomo.common.dependencies import networkx as nx
def get_scc_of_projection(graph, top_nodes, matching=None):
    """Return the topologically ordered strongly connected components of a
    bipartite graph, projected with respect to a perfect matching

    The provided undirected bipartite graph is projected into a directed graph
    on the set of "top nodes" by treating "matched edges" as out-edges and
    "unmatched edges" as in-edges. Then the strongly connected components of
    the directed graph are computed. These strongly connected components are
    unique, regardless of the choice of perfect matching. The strongly connected
    components form a directed acyclic graph, and are returned in a topological
    order. The order is unique, as ambiguities are resolved "lexicographically".

    The "direction" of the projection (where matched edges are out-edges)
    leads to a block *lower* triangular permutation when the top nodes
    correspond to *rows* in the bipartite graph of a matrix.

    Parameters
    ----------
    graph: NetworkX Graph
        A bipartite graph
    top_nodes: list
        One of the bipartite sets in the graph
    matching: dict
        Maps each node in ``top_nodes`` to its matched node

    Returns
    -------
    list of lists
        The outer list is a list of strongly connected components. Each
        strongly connected component is a list of tuples of matched nodes.
        The first node is a "top node", and the second is an "other node".

    """
    nxb = nx.algorithms.bipartite
    nxd = nx.algorithms.dag
    if not nxb.is_bipartite(graph):
        raise RuntimeError('Provided graph is not bipartite.')
    M = len(top_nodes)
    N = len(graph.nodes) - M
    if M != N:
        raise RuntimeError('get_scc_of_projection does not support bipartite graphs with bipartite sets of different cardinalities. Got sizes %s and %s.' % (M, N))
    if matching is None:
        matching = nxb.maximum_matching(graph, top_nodes=top_nodes)
    if len(matching) != 2 * M:
        raise RuntimeError('get_scc_of_projection does not support bipartite graphs without a perfect matching. Got a graph with %s nodes per bipartite set and a matching of cardinality %s.' % (M, len(matching) / 2))
    scc_list, dag = _get_scc_dag_of_projection(graph, top_nodes, matching)
    scc_order = list(nxd.lexicographical_topological_sort(dag))
    ordered_node_subsets = [sorted([(i, matching[i]) for i in scc_list[scc_idx]]) for scc_idx in scc_order]
    return ordered_node_subsets