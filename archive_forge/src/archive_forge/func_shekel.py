import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def shekel(individual, a, c):
    """The Shekel multimodal function can have any number of maxima. The number
    of maxima is given by the length of any of the arguments *a* or *c*, *a*
    is a matrix of size :math:`M\\times N`, where *M* is the number of maxima
    and *N* the number of dimensions and *c* is a :math:`M\\times 1` vector.

    :math:`f_\\text{Shekel}(\\mathbf{x}) = \\sum_{i = 1}^{M} \\frac{1}{c_{i} +
    \\sum_{j = 1}^{N} (x_{j} - a_{ij})^2 }`

    The following figure uses

    :math:`\\mathcal{A} = \\begin{bmatrix} 0.5 & 0.5 \\\\ 0.25 & 0.25 \\\\
    0.25 & 0.75 \\\\ 0.75 & 0.25 \\\\ 0.75 & 0.75 \\end{bmatrix}` and
    :math:`\\mathbf{c} = \\begin{bmatrix} 0.002 \\\\ 0.005 \\\\ 0.005
    \\\\ 0.005 \\\\ 0.005 \\end{bmatrix}`, thus defining 5 maximums in
    :math:`\\mathbb{R}^2`.

    .. plot:: code/benchmarks/shekel.py
        :width: 67 %
    """
    return (sum((1.0 / (c[i] + sum(((individual[j] - aij) ** 2 for j, aij in enumerate(a[i])))) for i in range(len(c)))),)