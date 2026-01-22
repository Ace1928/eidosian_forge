from typing import Callable, List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
def exprval_in_vec_ineq(expr, vec):
    assert len(expr.shape) == 1
    n_entries = expr.shape[0]
    vec = np.sort(vec)
    d = np.diff(vec)
    repeated_d = np.broadcast_to(d, (n_entries, len(d)))
    z = Variable(shape=repeated_d.shape, boolean=True)
    main_con = expr == vec[0] + cp.sum(cp.multiply(repeated_d, z), axis=1)
    if d.size > 1:
        aux_cons = [z[:, 1:] <= z[:, :-1]]
    else:
        aux_cons = []
    return (main_con, aux_cons)