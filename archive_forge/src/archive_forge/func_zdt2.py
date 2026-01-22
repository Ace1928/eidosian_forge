import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def zdt2(individual):
    """ZDT2 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + \\frac{9}{n-1}\\sum_{i=2}^n x_i`

    :math:`f_{\\text{ZDT2}1}(\\mathbf{x}) = x_1`

    :math:`f_{\\text{ZDT2}2}(\\mathbf{x}) = g(\\mathbf{x})\\left[1 - \\left(\\frac{x_1}{g(\\mathbf{x})}\\right)^2\\right]`

    """
    g = 1.0 + 9.0 * sum(individual[1:]) / (len(individual) - 1)
    f1 = individual[0]
    f2 = g * (1 - (f1 / g) ** 2)
    return (f1, f2)