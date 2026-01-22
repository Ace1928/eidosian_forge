def print_node(node, assumptions=True):
    """
    Returns information about the "node".

    This includes class name, string representation and assumptions.

    Parameters
    ==========

    assumptions : bool, optional
        See the ``assumptions`` keyword in ``tree``
    """
    s = '%s: %s\n' % (node.__class__.__name__, str(node))
    if assumptions:
        d = node._assumptions
    else:
        d = None
    if d:
        for a in sorted(d):
            v = d[a]
            if v is None:
                continue
            s += '%s: %s\n' % (a, v)
    return s