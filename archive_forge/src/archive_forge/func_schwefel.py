import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def schwefel(individual):
    """Schwefel test objective function.

    .. list-table::
       :widths: 10 50
       :stub-columns: 1

       * - Type
         - minimization
       * - Range
         - :math:`x_i \\in [-500, 500]`
       * - Global optima
         - :math:`x_i = 420.96874636, \\forall i \\in \\lbrace 1 \\ldots N\\rbrace`, :math:`f(\\mathbf{x}) = 0`
       * - Function
         - :math:`f(\\mathbf{x}) = 418.9828872724339\\cdot N - \\
            \\sum_{i=1}^N\\,x_i\\sin\\left(\\sqrt{|x_i|}\\right)`


    .. plot:: code/benchmarks/schwefel.py
        :width: 67 %
    """
    N = len(individual)
    return (418.9828872724339 * N - sum((x * sin(sqrt(abs(x))) for x in individual)),)