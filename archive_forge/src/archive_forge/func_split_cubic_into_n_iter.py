import math
from .errors import Error as Cu2QuError, ApproxNotFoundError
@cython.cfunc
@cython.inline
@cython.locals(p0=cython.complex, p1=cython.complex, p2=cython.complex, p3=cython.complex)
def split_cubic_into_n_iter(p0, p1, p2, p3, n):
    """Split a cubic Bezier into n equal parts.

    Splits the curve into `n` equal parts by curve time.
    (t=0..1/n, t=1/n..2/n, ...)

    Args:
        p0 (complex): Start point of curve.
        p1 (complex): First handle of curve.
        p2 (complex): Second handle of curve.
        p3 (complex): End point of curve.

    Returns:
        An iterator yielding the control points (four complex values) of the
        subcurves.
    """
    if n == 2:
        return iter(split_cubic_into_two(p0, p1, p2, p3))
    if n == 3:
        return iter(split_cubic_into_three(p0, p1, p2, p3))
    if n == 4:
        a, b = split_cubic_into_two(p0, p1, p2, p3)
        return iter(split_cubic_into_two(a[0], a[1], a[2], a[3]) + split_cubic_into_two(b[0], b[1], b[2], b[3]))
    if n == 6:
        a, b = split_cubic_into_two(p0, p1, p2, p3)
        return iter(split_cubic_into_three(a[0], a[1], a[2], a[3]) + split_cubic_into_three(b[0], b[1], b[2], b[3]))
    return _split_cubic_into_n_gen(p0, p1, p2, p3, n)