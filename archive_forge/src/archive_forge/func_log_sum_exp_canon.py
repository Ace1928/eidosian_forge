import numpy as np
from cvxpy.atoms import exp, promote, reshape, sum
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.dcp2cone.canonicalizers.exp_canon import exp_canon
def log_sum_exp_canon(expr, args):
    x = args[0]
    shape = expr.shape
    axis = expr.axis
    keepdims = expr.keepdims
    t = Variable(shape)
    if axis is None:
        promoted_t = promote(t, x.shape)
    elif axis == 0:
        promoted_t = Constant(np.ones((x.shape[0], 1))) @ reshape(t, (1,) + x.shape[1:])
    else:
        promoted_t = reshape(t, x.shape[:-1] + (1,)) @ Constant(np.ones((1, x.shape[1])))
    exp_expr = exp(x - promoted_t)
    obj, constraints = exp_canon(exp_expr, exp_expr.args)
    obj = sum(obj, axis=axis, keepdims=keepdims)
    ones = Constant(np.ones(shape))
    constraints.append(obj <= ones)
    return (t, constraints)