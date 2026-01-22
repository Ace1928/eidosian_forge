from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(graphs=None)
def from_prufer_sequence(sequence):
    """Returns the tree corresponding to the given Prüfer sequence.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    sequence : list
        A Prüfer sequence, which is a list of *n* - 2 integers between
        zero and *n* - 1, inclusive.

    Returns
    -------
    NetworkX graph
        The tree corresponding to the given Prüfer sequence.

    Raises
    ------
    NetworkXError
        If the Prüfer sequence is not valid.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence` function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`to_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    n = len(sequence) + 2
    degree = Counter(chain(sequence, range(n)))
    T = nx.empty_graph(n)
    not_orphaned = set()
    index = u = next((k for k in range(n) if degree[k] == 1))
    for v in sequence:
        if v < 0 or v > n - 1:
            raise nx.NetworkXError(f'Invalid Prufer sequence: Values must be between 0 and {n - 1}, got {v}')
        T.add_edge(u, v)
        not_orphaned.add(u)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next((k for k in range(index + 1, n) if degree[k] == 1))
    orphans = set(T) - not_orphaned
    u, v = orphans
    T.add_edge(u, v)
    return T