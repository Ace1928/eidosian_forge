import networkx as nx
from networkx.utils.decorators import not_implemented_for
@nx._dispatch(graphs={'t1': 0, 't2': 2})
def rooted_tree_isomorphism(t1, root1, t2, root2):
    """
    Given two rooted trees `t1` and `t2`,
    with roots `root1` and `root2` respectively
    this routine will determine if they are isomorphic.

    These trees may be either directed or undirected,
    but if they are directed, all edges should flow from the root.

    It returns the isomorphism, a mapping of the nodes of `t1` onto the nodes
    of `t2`, such that two trees are then identical.

    Note that two trees may have more than one isomorphism, and this
    routine just returns one valid mapping.

    Parameters
    ----------
    `t1` :  NetworkX graph
        One of the trees being compared

    `root1` : a node of `t1` which is the root of the tree

    `t2` : undirected NetworkX graph
        The other tree being compared

    `root2` : a node of `t2` which is the root of the tree

    This is a subroutine used to implement `tree_isomorphism`, but will
    be somewhat faster if you already have rooted trees.

    Returns
    -------
    isomorphism : list
        A list of pairs in which the left element is a node in `t1`
        and the right element is a node in `t2`.  The pairs are in
        arbitrary order.  If the nodes in one tree is mapped to the names in
        the other, then trees will be identical. Note that an isomorphism
        will not necessarily be unique.

        If `t1` and `t2` are not isomorphic, then it returns the empty list.
    """
    assert nx.is_tree(t1)
    assert nx.is_tree(t2)
    dT, namemap, newroot1, newroot2 = root_trees(t1, root1, t2, root2)
    levels = assign_levels(dT, 0)
    h = max(levels.values())
    L = group_by_levels(levels)
    label = {v: 0 for v in dT}
    ordered_labels = {v: () for v in dT}
    ordered_children = {v: () for v in dT}
    for i in range(h - 1, 0, -1):
        for v in L[i]:
            if dT.out_degree(v) > 0:
                s = sorted(((label[u], u) for u in dT.successors(v)))
                ordered_labels[v], ordered_children[v] = list(zip(*s))
        forlabel = sorted(((ordered_labels[v], v) for v in L[i]))
        current = 0
        for i, (ol, v) in enumerate(forlabel):
            if i != 0 and ol != forlabel[i - 1][0]:
                current += 1
            label[v] = current
    isomorphism = []
    if label[newroot1] == 0 and label[newroot2] == 0:
        generate_isomorphism(newroot1, newroot2, isomorphism, ordered_children)
        isomorphism = [(namemap[u], namemap[v]) for u, v in isomorphism]
    return isomorphism