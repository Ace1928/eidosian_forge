import logging
import math
from pyomo.common.dependencies import numpy, numpy_available
from pyomo.common.collections import ComponentSet
from pyomo.core.base import Transformation, TransformationFactory
from pyomo.core import Var, ConstraintList, Expression, Objective
from pyomo.dae import ContinuousSet, DerivativeVar, Integral
from pyomo.dae.misc import generate_finite_elements
from pyomo.dae.misc import generate_colloc_points
from pyomo.dae.misc import expand_components
from pyomo.dae.misc import create_partial_expression
from pyomo.dae.misc import add_discretization_equations
from pyomo.dae.misc import add_continuity_equations
from pyomo.dae.misc import block_fully_discretized
from pyomo.dae.misc import get_index_information
from pyomo.dae.diffvar import DAE_Error
from pyomo.common.config import ConfigBlock, ConfigValue, PositiveInt, In
def calc_cp(alpha, beta, k):
    gamma = []
    factorial = math.factorial
    for i in range(k + 1):
        num = factorial(alpha + k) * factorial(alpha + beta + k + i)
        denom = factorial(alpha + i) * factorial(k - i) * factorial(i)
        gamma.insert(i, num / denom)
    poly = []
    for i in range(k + 1):
        if i == 0:
            poly.insert(i, gamma[i])
        else:
            prod = [1]
            j = 1
            while j <= i:
                prod = conv(prod, [1, -1])
                j += 1
            while len(poly) < len(prod):
                poly.insert(0, 0)
            prod = [gamma[i] * t for t in prod]
            poly = [sum(pair) for pair in zip(poly, prod)]
    cp = numpy.roots(poly)
    return numpy.sort(cp).tolist()