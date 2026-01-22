from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch(graphs='T')
def to_nested_tuple(T, root, canonical_form=False):
    """Returns a nested tuple representation of the given tree.

    The nested tuple representation of a tree is defined
    recursively. The tree with one node and no edges is represented by
    the empty tuple, ``()``. A tree with ``k`` subtrees is represented
    by a tuple of length ``k`` in which each element is the nested tuple
    representation of a subtree.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    root : node
        The node in ``T`` to interpret as the root of the tree.

    canonical_form : bool
        If ``True``, each tuple is sorted so that the function returns
        a canonical form for rooted trees. This means "lighter" subtrees
        will appear as nested tuples before "heavier" subtrees. In this
        way, each isomorphic rooted tree has the same nested tuple
        representation.

    Returns
    -------
    tuple
        A nested tuple representation of the tree.

    Notes
    -----
    This function is *not* the inverse of :func:`from_nested_tuple`; the
    only guarantee is that the rooted trees are isomorphic.

    See also
    --------
    from_nested_tuple
    to_prufer_sequence

    Examples
    --------
    The tree need not be a balanced binary tree::

        >>> T = nx.Graph()
        >>> T.add_edges_from([(0, 1), (0, 2), (0, 3)])
        >>> T.add_edges_from([(1, 4), (1, 5)])
        >>> T.add_edges_from([(3, 6), (3, 7)])
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        (((), ()), (), ((), ()))

    Continuing the above example, if ``canonical_form`` is ``True``, the
    nested tuples will be sorted::

        >>> nx.to_nested_tuple(T, root, canonical_form=True)
        ((), ((), ()), ((), ()))

    Even the path graph can be interpreted as a tree::

        >>> T = nx.path_graph(4)
        >>> root = 0
        >>> nx.to_nested_tuple(T, root)
        ((((),),),)

    """

    def _make_tuple(T, root, _parent):
        """Recursively compute the nested tuple representation of the
        given rooted tree.

        ``_parent`` is the parent node of ``root`` in the supertree in
        which ``T`` is a subtree, or ``None`` if ``root`` is the root of
        the supertree. This argument is used to determine which
        neighbors of ``root`` are children and which is the parent.

        """
        children = set(T[root]) - {_parent}
        if len(children) == 0:
            return ()
        nested = (_make_tuple(T, v, root) for v in children)
        if canonical_form:
            nested = sorted(nested)
        return tuple(nested)
    if not nx.is_tree(T):
        raise nx.NotATree('provided graph is not a tree')
    if root not in T:
        raise nx.NodeNotFound(f'Graph {T} contains no node {root}')
    return _make_tuple(T, root, None)