import logging
import random
from pyomo.core import Var
def simple_midpoint(val, lb, ub):
    return (lb + ub) * 0.5