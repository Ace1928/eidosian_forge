import numpy as np
from cvxpy import Constant
from cvxpy.atoms.affine.binary_operators import outer
from cvxpy.atoms.affine.sum import sum
from cvxpy.atoms.affine.vec import vec
from cvxpy.expressions.variable import Variable
def dotsort_canon(expr, args):
    x = args[0]
    w = args[1]
    if isinstance(w, Constant):
        w_unique, w_counts = np.unique(w.value, return_counts=True)
    else:
        w_unique, w_counts = (w, np.ones(w.size))
    t = Variable((x.size, 1), nonneg=True)
    q = Variable((1, w_unique.size))
    obj = sum(t) + q @ w_counts
    x_w_unique_outer_product = outer(vec(x), vec(w_unique))
    constraints = [x_w_unique_outer_product <= t + q]
    return (obj, constraints)