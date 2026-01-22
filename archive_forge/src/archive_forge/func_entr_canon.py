import numpy as np
from cvxpy.constraints.exponential import ExpCone
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
def entr_canon(expr, args):
    x = args[0]
    shape = expr.shape
    t = Variable(shape)
    ones = Constant(np.ones(shape))
    constraints = [ExpCone(t, x, ones)]
    return (t, constraints)