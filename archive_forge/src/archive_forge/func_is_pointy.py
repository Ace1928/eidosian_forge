from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def is_pointy(a, b, c, threshold=QQ('3/4')):
    """
    Whether the angle (a, b, c) is less than pi/6.  This function is
    not currently used by anything.

    >>> p = Vector3([ 0, 0, 1])
    >>> q = Vector3([10, 0, 1])
    >>> r = Vector3([0, 10, 1])
    >>> s = Vector3([9, 1, 1])
    >>> t = Vector3([-9, 1, 1])
    >>> is_pointy(t, p, q)
    False
    >>> is_pointy(q, p, r)
    False
    >>> is_pointy(p, q, r)
    False
    >>> is_pointy(s, p, q)
    True
    >>> is_pointy(q, p, q)
    True
    """
    v = a - b
    w = c - b
    dot = v * w
    if dot <= 0:
        return False
    norms = v * v * (w * w)
    return dot ** 2 / norms >= threshold