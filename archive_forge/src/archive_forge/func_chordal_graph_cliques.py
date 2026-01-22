import sys
import networkx as nx
from networkx.algorithms.components import connected_components
from networkx.utils import arbitrary_element, not_implemented_for
@nx._dispatch
def chordal_graph_cliques(G):
    """Returns all maximal cliques of a chordal graph.

    The algorithm breaks the graph in connected components and performs a
    maximum cardinality search in each component to get the cliques.

    Parameters
    ----------
    G : graph
      A NetworkX graph

    Yields
    ------
    frozenset of nodes
        Maximal cliques, each of which is a frozenset of
        nodes in `G`. The order of cliques is arbitrary.

    Raises
    ------
    NetworkXError
        The algorithm does not support DiGraph, MultiGraph and MultiDiGraph.
        The algorithm can only be applied to chordal graphs. If the input
        graph is found to be non-chordal, a :exc:`NetworkXError` is raised.

    Examples
    --------
    >>> e = [
    ...     (1, 2),
    ...     (1, 3),
    ...     (2, 3),
    ...     (2, 4),
    ...     (3, 4),
    ...     (3, 5),
    ...     (3, 6),
    ...     (4, 5),
    ...     (4, 6),
    ...     (5, 6),
    ...     (7, 8),
    ... ]
    >>> G = nx.Graph(e)
    >>> G.add_node(9)
    >>> cliques = [c for c in chordal_graph_cliques(G)]
    >>> cliques[0]
    frozenset({1, 2, 3})
    """
    for C in (G.subgraph(c).copy() for c in connected_components(G)):
        if C.number_of_nodes() == 1:
            if nx.number_of_selfloops(C) > 0:
                raise nx.NetworkXError('Input graph is not chordal.')
            yield frozenset(C.nodes())
        else:
            unnumbered = set(C.nodes())
            v = arbitrary_element(C)
            unnumbered.remove(v)
            numbered = {v}
            clique_wanna_be = {v}
            while unnumbered:
                v = _max_cardinality_node(C, unnumbered, numbered)
                unnumbered.remove(v)
                numbered.add(v)
                new_clique_wanna_be = set(C.neighbors(v)) & numbered
                sg = C.subgraph(clique_wanna_be)
                if _is_complete_graph(sg):
                    new_clique_wanna_be.add(v)
                    if not new_clique_wanna_be >= clique_wanna_be:
                        yield frozenset(clique_wanna_be)
                    clique_wanna_be = new_clique_wanna_be
                else:
                    raise nx.NetworkXError('Input graph is not chordal.')
            yield frozenset(clique_wanna_be)