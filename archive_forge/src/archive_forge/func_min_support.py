from .rational_linear_algebra import Matrix, Vector2, Vector3, QQ, rational_sqrt
def min_support(v):
    """
    >>> min_support(Vector3([0, 0, -1]))
    2
    """
    for i, e in enumerate(v):
        if e != 0:
            return i
    raise ValueError('Vector is 0')