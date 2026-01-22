from typing import Callable, List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.constraints import FiniteSet
from cvxpy.constraints.constraint import Constraint
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
def exprval_in_vec_eq(expr, vec):
    assert len(expr.shape) == 1
    n_entries = expr.shape[0]
    repeated_vec = np.broadcast_to(vec, (n_entries, len(vec)))
    z = Variable(repeated_vec.shape, boolean=True)
    main_con = cp.sum(cp.multiply(repeated_vec, z), axis=1) == expr
    aux_cons = [cp.sum(z, axis=1) == 1]
    return (main_con, aux_cons)