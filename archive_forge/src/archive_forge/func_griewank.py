import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def griewank(individual):
    """Griewank test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-600, 600]`
       * - Global optima
         - :math:`x_i = 0, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = \\frac{1}{4000}\\sum_{i=1}^N\\,x_i^2 - \\
                  \\prod_{i=1}^N\\cos\\left(\\frac{x_i}{\\sqrt{i}}\\right) + 1`

    .. plot:: code/benchmarks/griewank.py
       :width: 67 %
    """
    return (1.0 / 4000.0 * sum((x ** 2 for x in individual)) - reduce(mul, (cos(x / sqrt(i + 1.0)) for i, x in enumerate(individual)), 1) + 1,)