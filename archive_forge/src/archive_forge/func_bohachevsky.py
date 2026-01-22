import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def bohachevsky(individual):
    """Bohachevsky test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-100, 100]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         -  :math:`f(\\mathbf{x}) = \\sum_{i=1}^{N-1}(x_i^2 + 2x_{i+1}^2 - \\
                   0.3\\cos(3\\pi x_i) - 0.4\\cos(4\\pi x_{i+1}) + 0.7)`

    .. plot:: code/benchmarks/bohachevsky.py
       :width: 67 %
    """
    return (sum((x ** 2 + 2 * x1 ** 2 - 0.3 * cos(3 * pi * x) - 0.4 * cos(4 * pi * x1) + 0.7 for x, x1 in zip(individual[:-1], individual[1:]))),)