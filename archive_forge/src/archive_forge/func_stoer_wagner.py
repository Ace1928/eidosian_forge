from itertools import islice
import networkx as nx
from ...utils import BinaryHeap, arbitrary_element, not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatch(edge_attrs='weight')
def stoer_wagner(G, weight='weight', heap=BinaryHeap):
    """Returns the weighted minimum edge cut using the Stoer-Wagner algorithm.

    Determine the minimum edge cut of a connected graph using the
    Stoer-Wagner algorithm. In weighted cases, all weights must be
    nonnegative.

    The running time of the algorithm depends on the type of heaps used:

    ============== =============================================
    Type of heap   Running time
    ============== =============================================
    Binary heap    $O(n (m + n) \\log n)$
    Fibonacci heap $O(nm + n^2 \\log n)$
    Pairing heap   $O(2^{2 \\sqrt{\\log \\log n}} nm + n^2 \\log n)$
    ============== =============================================

    Parameters
    ----------
    G : NetworkX graph
        Edges of the graph are expected to have an attribute named by the
        weight parameter below. If this attribute is not present, the edge is
        considered to have unit weight.

    weight : string
        Name of the weight attribute of the edges. If the attribute is not
        present, unit weight is assumed. Default value: 'weight'.

    heap : class
        Type of heap to be used in the algorithm. It should be a subclass of
        :class:`MinHeap` or implement a compatible interface.

        If a stock heap implementation is to be used, :class:`BinaryHeap` is
        recommended over :class:`PairingHeap` for Python implementations without
        optimized attribute accesses (e.g., CPython) despite a slower
        asymptotic running time. For Python implementations with optimized
        attribute accesses (e.g., PyPy), :class:`PairingHeap` provides better
        performance. Default value: :class:`BinaryHeap`.

    Returns
    -------
    cut_value : integer or float
        The sum of weights of edges in a minimum cut.

    partition : pair of node lists
        A partitioning of the nodes that defines a minimum cut.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or a multigraph.

    NetworkXError
        If the graph has less than two nodes, is not connected or has a
        negative-weighted edge.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edge("x", "a", weight=3)
    >>> G.add_edge("x", "b", weight=1)
    >>> G.add_edge("a", "c", weight=3)
    >>> G.add_edge("b", "c", weight=5)
    >>> G.add_edge("b", "d", weight=4)
    >>> G.add_edge("d", "e", weight=2)
    >>> G.add_edge("c", "y", weight=2)
    >>> G.add_edge("e", "y", weight=3)
    >>> cut_value, partition = nx.stoer_wagner(G)
    >>> cut_value
    4
    """
    n = len(G)
    if n < 2:
        raise nx.NetworkXError('graph has less than two nodes.')
    if not nx.is_connected(G):
        raise nx.NetworkXError('graph is not connected.')
    G = nx.Graph(((u, v, {'weight': e.get(weight, 1)}) for u, v, e in G.edges(data=True) if u != v))
    for u, v, e in G.edges(data=True):
        if e['weight'] < 0:
            raise nx.NetworkXError('graph has a negative-weighted edge.')
    cut_value = float('inf')
    nodes = set(G)
    contractions = []
    for i in range(n - 1):
        u = arbitrary_element(G)
        A = {u}
        h = heap()
        for v, e in G[u].items():
            h.insert(v, -e['weight'])
        for j in range(n - i - 2):
            u = h.pop()[0]
            A.add(u)
            for v, e in G[u].items():
                if v not in A:
                    h.insert(v, h.get(v, 0) - e['weight'])
        v, w = h.min()
        w = -w
        if w < cut_value:
            cut_value = w
            best_phase = i
        contractions.append((u, v))
        for w, e in G[v].items():
            if w != u:
                if w not in G[u]:
                    G.add_edge(u, w, weight=e['weight'])
                else:
                    G[u][w]['weight'] += e['weight']
        G.remove_node(v)
    G = nx.Graph(islice(contractions, best_phase))
    v = contractions[best_phase][1]
    G.add_node(v)
    reachable = set(nx.single_source_shortest_path_length(G, v))
    partition = (list(reachable), list(nodes - reachable))
    return (cut_value, partition)