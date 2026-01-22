import numpy as np
from cvxpy.expressions.constants.constant import Constant
def constant_canon(expr, args):
    del args
    return (Constant(np.log(expr.value)), [])