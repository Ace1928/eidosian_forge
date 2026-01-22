import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def zdt6(individual):
    """ZDT6 multiobjective function.

    :math:`g(\\mathbf{x}) = 1 + 9 \\left[ \\left(\\sum_{i=2}^n x_i\\right)/(n-1) \\right]^{0.25}`

    :math:`f_{\\text{ZDT6}1}(\\mathbf{x}) = 1 - \\exp(-4x_1)\\sin^6(6\\pi x_1)`

    :math:`f_{\\text{ZDT6}2}(\\mathbf{x}) = g(\\mathbf{x}) \\left[ 1 - (f_{\\text{ZDT6}1}(\\mathbf{x})/g(\\mathbf{x}))^2 \\right]`

    """
    g = 1 + 9 * (sum(individual[1:]) / (len(individual) - 1)) ** 0.25
    f1 = 1 - exp(-4 * individual[0]) * sin(6 * pi * individual[0]) ** 6
    f2 = g * (1 - (f1 / g) ** 2)
    return (f1, f2)