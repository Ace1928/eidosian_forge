import random
from math import sin, cos, pi, exp, e, sqrt
from operator import mul
from functools import reduce
def poloni(individual):
    """Poloni's multiobjective function on a two attribute *individual*. From:
    C. Poloni, "Hybrid GA for multi objective aerodynamic shape optimization",
    in Genetic Algorithms in Engineering and Computer Science, 1997.

    :math:`A_1 = 0.5 \\sin (1) - 2 \\cos (1) + \\sin (2) - 1.5 \\cos (2)`

    :math:`A_2 = 1.5 \\sin (1) - \\cos (1) + 2 \\sin (2) - 0.5 \\cos (2)`

    :math:`B_1 = 0.5 \\sin (x_1) - 2 \\cos (x_1) + \\sin (x_2) - 1.5 \\cos (x_2)`

    :math:`B_2 = 1.5 \\sin (x_1) - cos(x_1) + 2 \\sin (x_2) - 0.5 \\cos (x_2)`

    :math:`f_{\\text{Poloni}1}(\\mathbf{x}) = 1 + (A_1 - B_1)^2 + (A_2 - B_2)^2`

    :math:`f_{\\text{Poloni}2}(\\mathbf{x}) = (x_1 + 3)^2 + (x_2 + 1)^2`
    """
    x_1 = individual[0]
    x_2 = individual[1]
    A_1 = 0.5 * sin(1) - 2 * cos(1) + sin(2) - 1.5 * cos(2)
    A_2 = 1.5 * sin(1) - cos(1) + 2 * sin(2) - 0.5 * cos(2)
    B_1 = 0.5 * sin(x_1) - 2 * cos(x_1) + sin(x_2) - 1.5 * cos(x_2)
    B_2 = 1.5 * sin(x_1) - cos(x_1) + 2 * sin(x_2) - 0.5 * cos(x_2)
    return (1 + (A_1 - B_1) ** 2 + (A_2 - B_2) ** 2, (x_1 + 3) ** 2 + (x_2 + 1) ** 2)