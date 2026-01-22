import math
import numbers
from functools import reduce
import networkx as nx
from networkx.utils import nodes_or_number, py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def preferential_attachment_graph(aseq, p, create_using=None, seed=None):
    """Create a bipartite graph with a preferential attachment model from
    a given single degree sequence.

    The graph is composed of two partitions. Set A has nodes 0 to
    (len(aseq) - 1) and set B has nodes starting with node len(aseq).
    The number of nodes in set B is random.

    Parameters
    ----------
    aseq : list
       Degree sequence for node set A.
    p :  float
       Probability that a new bottom node is added.
    create_using : NetworkX graph instance, optional
       Return graph of this type.
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    References
    ----------
    .. [1] Guillaume, J.L. and Latapy, M.,
       Bipartite graphs as models of complex networks.
       Physica A: Statistical Mechanics and its Applications,
       2006, 371(2), pp.795-813.
    .. [2] Jean-Loup Guillaume and Matthieu Latapy,
       Bipartite structure of all complex networks,
       Inf. Process. Lett. 90, 2004, pg. 215-221
       https://doi.org/10.1016/j.ipl.2004.03.007

    Notes
    -----
    The nodes are assigned the attribute 'bipartite' with the value 0 or 1
    to indicate which bipartite set the node belongs to.

    This function is not imported in the main namespace.
    To use it use nx.bipartite.preferential_attachment_graph
    """
    G = nx.empty_graph(0, create_using, default=nx.MultiGraph)
    if G.is_directed():
        raise nx.NetworkXError('Directed Graph not supported')
    if p > 1:
        raise nx.NetworkXError(f'probability {p} > 1')
    naseq = len(aseq)
    G = _add_nodes_with_bipartite_label(G, naseq, 0)
    vv = [[v] * aseq[v] for v in range(naseq)]
    while vv:
        while vv[0]:
            source = vv[0][0]
            vv[0].remove(source)
            if seed.random() < p or len(G) == naseq:
                target = len(G)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
            else:
                bb = [[b] * G.degree(b) for b in range(naseq, len(G))]
                bbstubs = reduce(lambda x, y: x + y, bb)
                target = seed.choice(bbstubs)
                G.add_node(target, bipartite=1)
                G.add_edge(source, target)
        vv.remove(vv[0])
    G.name = 'bipartite_preferential_attachment_model'
    return G