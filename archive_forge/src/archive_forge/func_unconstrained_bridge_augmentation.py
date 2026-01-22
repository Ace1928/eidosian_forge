import itertools as it
import math
from collections import defaultdict, namedtuple
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@nx._dispatch
def unconstrained_bridge_augmentation(G):
    """Finds an optimal 2-edge-augmentation of G using the fewest edges.

    This is an implementation of the algorithm detailed in [1]_.
    The basic idea is to construct a meta-graph of bridge-ccs, connect leaf
    nodes of the trees to connect the entire graph, and finally connect the
    leafs of the tree in dfs-preorder to bridge connect the entire graph.

    Parameters
    ----------
    G : NetworkX graph
       An undirected graph.

    Yields
    ------
    edge : tuple
        Edges in the bridge augmentation of G

    Notes
    -----
    Input: a graph G.
    First find the bridge components of G and collapse each bridge-cc into a
    node of a metagraph graph C, which is guaranteed to be a forest of trees.

    C contains p "leafs" --- nodes with exactly one incident edge.
    C contains q "isolated nodes" --- nodes with no incident edges.

    Theorem: If p + q > 1, then at least :math:`ceil(p / 2) + q` edges are
        needed to bridge connect C. This algorithm achieves this min number.

    The method first adds enough edges to make G into a tree and then pairs
    leafs in a simple fashion.

    Let n be the number of trees in C. Let v(i) be an isolated vertex in the
    i-th tree if one exists, otherwise it is a pair of distinct leafs nodes
    in the i-th tree. Alternating edges from these sets (i.e.  adding edges
    A1 = [(v(i)[0], v(i + 1)[1]), v(i + 1)[0], v(i + 2)[1])...]) connects C
    into a tree T. This tree has p' = p + 2q - 2(n -1) leafs and no isolated
    vertices. A1 has n - 1 edges. The next step finds ceil(p' / 2) edges to
    biconnect any tree with p' leafs.

    Convert T into an arborescence T' by picking an arbitrary root node with
    degree >= 2 and directing all edges away from the root. Note the
    implementation implicitly constructs T'.

    The leafs of T are the nodes with no existing edges in T'.
    Order the leafs of T' by DFS preorder. Then break this list in half
    and add the zipped pairs to A2.

    The set A = A1 + A2 is the minimum augmentation in the metagraph.

    To convert this to edges in the original graph

    References
    ----------
    .. [1] Eswaran, Kapali P., and R. Endre Tarjan. (1975) Augmentation problems.
        http://epubs.siam.org/doi/abs/10.1137/0205044

    See Also
    --------
    :func:`bridge_augmentation`
    :func:`k_edge_augmentation`

    Examples
    --------
    >>> G = nx.path_graph((1, 2, 3, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 7)]
    >>> G = nx.path_graph((1, 2, 3, 2, 4, 5, 6, 7))
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 3), (3, 7)]
    >>> G = nx.Graph([(0, 1), (0, 2), (1, 2)])
    >>> G.add_node(4)
    >>> sorted(unconstrained_bridge_augmentation(G))
    [(1, 4), (4, 0)]
    """
    bridge_ccs = list(nx.connectivity.bridge_components(G))
    C = collapse(G, bridge_ccs)
    vset1 = [tuple(cc) * 2 if len(cc) == 1 else sorted(cc, key=C.degree)[0:2] for cc in nx.connected_components(C)]
    if len(vset1) > 1:
        nodes1 = [vs[0] for vs in vset1]
        nodes2 = [vs[1] for vs in vset1]
        A1 = list(zip(nodes1[1:], nodes2))
    else:
        A1 = []
    T = C.copy()
    T.add_edges_from(A1)
    leafs = [n for n, d in T.degree() if d == 1]
    if len(leafs) == 1:
        A2 = []
    if len(leafs) == 2:
        A2 = [tuple(leafs)]
    else:
        try:
            root = next((n for n, d in T.degree() if d > 1))
        except StopIteration:
            return
        v2 = [n for n in nx.dfs_preorder_nodes(T, root) if T.degree(n) == 1]
        half = math.ceil(len(v2) / 2)
        A2 = list(zip(v2[:half], v2[-half:]))
    aug_tree_edges = A1 + A2
    inverse = defaultdict(list)
    for k, v in C.graph['mapping'].items():
        inverse[v].append(k)
    inverse = {mu: sorted(mapped, key=lambda u: (G.degree(u), u)) for mu, mapped in inverse.items()}
    G2 = G.copy()
    for mu, mv in aug_tree_edges:
        for u, v in it.product(inverse[mu], inverse[mv]):
            if not G2.has_edge(u, v):
                G2.add_edge(u, v)
                yield (u, v)
                break