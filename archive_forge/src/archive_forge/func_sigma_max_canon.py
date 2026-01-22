import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.bmat import bmat
from cvxpy.constraints.psd import PSD
from cvxpy.expressions.variable import Variable
def sigma_max_canon(expr, args):
    A = args[0]
    n, m = A.shape
    shape = expr.shape
    if not np.prod(shape) == 1:
        raise RuntimeError('Invalid shape of expr in sigma_max canonicalization.')
    t = Variable(shape)
    tI_n = sp.eye(n) * t
    tI_m = sp.eye(m) * t
    X = bmat([[tI_n, A], [A.T, tI_m]])
    constraints = [PSD(X)]
    return (t, constraints)