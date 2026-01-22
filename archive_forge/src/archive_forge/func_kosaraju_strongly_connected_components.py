import networkx as nx
from networkx.utils.decorators import not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def kosaraju_strongly_connected_components(G, source=None):
    """Generate nodes in strongly connected components of graph.

    Parameters
    ----------
    G : NetworkX Graph
        A directed graph.

    Returns
    -------
    comp : generator of sets
        A generator of sets of nodes, one for each strongly connected
        component of G.

    Raises
    ------
    NetworkXNotImplemented
        If G is undirected.

    Examples
    --------
    Generate a sorted list of strongly connected components, largest first.

    >>> G = nx.cycle_graph(4, create_using=nx.DiGraph())
    >>> nx.add_cycle(G, [10, 11, 12])
    >>> [
    ...     len(c)
    ...     for c in sorted(
    ...         nx.kosaraju_strongly_connected_components(G), key=len, reverse=True
    ...     )
    ... ]
    [4, 3]

    If you only want the largest component, it's more efficient to
    use max instead of sort.

    >>> largest = max(nx.kosaraju_strongly_connected_components(G), key=len)

    See Also
    --------
    strongly_connected_components

    Notes
    -----
    Uses Kosaraju's algorithm.

    """
    post = list(nx.dfs_postorder_nodes(G.reverse(copy=False), source=source))
    seen = set()
    while post:
        r = post.pop()
        if r in seen:
            continue
        c = nx.dfs_preorder_nodes(G, r)
        new = {v for v in c if v not in seen}
        seen.update(new)
        yield new