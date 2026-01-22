import logging
import random
from pyomo.core import Var
def rand_distributed(val, lb, ub, divisions=9):
    """Random choice among evenly distributed set of values between bounds."""
    set_distributed_vals = linspace(lb, ub, divisions)
    return random.choice(set_distributed_vals)