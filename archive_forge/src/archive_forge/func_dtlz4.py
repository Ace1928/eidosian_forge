import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def dtlz4(individual, obj, alpha):
    """DTLZ4 multiobjective function. It returns a tuple of *obj* values. The
    individual must have at least *obj* elements. The *alpha* parameter allows
    for a meta-variable mapping in :func:`dtlz2` :math:`x_i \\rightarrow
    x_i^\\alpha`, the authors suggest :math:`\\alpha = 100`.
    From: K. Deb, L. Thiele, M. Laumanns and E. Zitzler. Scalable Multi-Objective
    Optimization Test Problems. CEC 2002, p. 825 - 830, IEEE Press, 2002.

    :math:`g(\\mathbf{x}_m) = \\sum_{x_i \\in \\mathbf{x}_m} (x_i - 0.5)^2`

    :math:`f_{\\text{DTLZ4}1}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\prod_{i=1}^{m-1} \\cos(0.5x_i^\\alpha\\pi)`

    :math:`f_{\\text{DTLZ4}2}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{m-1}^\\alpha\\pi ) \\prod_{i=1}^{m-2} \\cos(0.5x_i^\\alpha\\pi)`

    :math:`\\ldots`

    :math:`f_{\\text{DTLZ4}m}(\\mathbf{x}) = (1 + g(\\mathbf{x}_m)) \\sin(0.5x_{1}^\\alpha\\pi )`

    Where :math:`m` is the number of objectives and :math:`\\mathbf{x}_m` is a
    vector of the remaining attributes :math:`[x_m~\\ldots~x_n]` of the
    individual in :math:`n > m` dimensions.
    """
    xc = individual[:obj - 1]
    xm = individual[obj - 1:]
    g = sum(((xi - 0.5) ** 2 for xi in xm))
    f = [(1.0 + g) * reduce(mul, (cos(0.5 * xi ** alpha * pi) for xi in xc), 1.0)]
    f.extend(((1.0 + g) * reduce(mul, (cos(0.5 * xi ** alpha * pi) for xi in xc[:m]), 1) * sin(0.5 * xc[m] ** alpha * pi) for m in range(obj - 2, -1, -1)))
    return f