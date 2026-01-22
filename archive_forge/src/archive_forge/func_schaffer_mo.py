import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def schaffer_mo(individual):
    """Schaffer's multiobjective function on a one attribute *individual*.
    From: J. D. Schaffer, "Multiple objective optimization with vector
    evaluated genetic algorithms", in Proceedings of the First International
    Conference on Genetic Algorithms, 1987.

    :math:`f_{\\text{Schaffer}1}(\\mathbf{x}) = x_1^2`

    :math:`f_{\\text{Schaffer}2}(\\mathbf{x}) = (x_1-2)^2`
    """
    return (individual[0] ** 2, (individual[0] - 2) ** 2)