from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def find_convex_combination(a, b, u):
    """
    >>> p = Vector2([0, 1])
    >>> q = Vector2([1, 0])
    >>> t = QQ(1)/5

    This next awkward cast is required when using the PARI kernel
    because of the permissive behavior of Gen.__mul__.

    >>> r = Vector2((1 - t)*p + t*q)
    >>> find_convex_combination(p, q, r)
    1/5
    >>> find_convex_combination(p, 3*p, 2*p)
    1/2
    """
    if a == b:
        raise ValueError('Segment is degenerate')
    v, w = (b - a, u - a)
    if not are_parallel(v, w):
        raise ValueError('Point not on line')
    i = min_support(v)
    t = w[i] / v[i]
    assert (1 - t) * a + t * b == u
    return t