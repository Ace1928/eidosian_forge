from collections import Counter, defaultdict, deque
import networkx as nx
from networkx.utils import groups, not_implemented_for, py_random_state
@py_random_state('seed')
@nx._dispatch(edge_attrs='weight')
def fast_label_propagation_communities(G, *, weight=None, seed=None):
    """Returns communities in `G` as detected by fast label propagation.

    The fast label propagation algorithm is described in [1]_. The algorithm is
    probabilistic and the found communities may vary in different executions.

    The algorithm operates as follows. First, the community label of each node is
    set to a unique label. The algorithm then repeatedly updates the labels of
    the nodes to the most frequent label in their neighborhood. In case of ties,
    a random label is chosen from the most frequent labels.

    The algorithm maintains a queue of nodes that still need to be processed.
    Initially, all nodes are added to the queue in a random order. Then the nodes
    are removed from the queue one by one and processed. If a node updates its label,
    all its neighbors that have a different label are added to the queue (if not
    already in the queue). The algorithm stops when the queue is empty.

    Parameters
    ----------
    G : Graph, DiGraph, MultiGraph, or MultiDiGraph
      Any NetworkX graph.

    weight : string, or None (default)
      The edge attribute representing a non-negative weight of an edge. If None,
      each edge is assumed to have weight one. The weight of an edge is used in
      determining the frequency with which a label appears among the neighbors of
      a node (edge with weight `w` is equivalent to `w` unweighted edges).

    seed : integer, random_state, or None (default)
      Indicator of random number generation state. See :ref:`Randomness<randomness>`.

    Returns
    -------
    communities : iterable
      Iterable of communities given as sets of nodes.

    Notes
    -----
    Edge directions are ignored for directed graphs.
    Edge weights must be non-negative numbers.

    References
    ----------
    .. [1] Vincent A. Traag & Lovro Šubelj. "Large network community detection by
    fast label propagation." Scientific Reports 13 (2023): 2701.
    https://doi.org/10.1038/s41598-023-29610-z
    """
    nodes_queue = deque(G)
    seed.shuffle(nodes_queue)
    nodes_set = set(G)
    comms = {node: i for i, node in enumerate(G)}
    while nodes_queue:
        node = nodes_queue.popleft()
        nodes_set.remove(node)
        if G.degree(node) > 0:
            label_freqs = _fast_label_count(G, comms, node, weight)
            max_freq = max(label_freqs.values())
            comm = seed.choice([comm for comm in label_freqs if label_freqs[comm] == max_freq])
            if comms[node] != comm:
                comms[node] = comm
                for nbr in nx.all_neighbors(G, node):
                    if comms[nbr] != comm and nbr not in nodes_set:
                        nodes_queue.append(nbr)
                        nodes_set.add(nbr)
    yield from groups(comms).values()