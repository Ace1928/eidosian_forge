from collections import defaultdict
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.cvxcore.python import canonInterface
def special_index_canon(expr, args):
    select_mat = expr._select_mat
    final_shape = expr._select_mat.shape
    select_vec = np.reshape(select_mat, select_mat.size, order='F')
    arg = args[0]
    identity = sp.eye(arg.size).tocsc()
    lowered = reshape(identity[select_vec] @ vec(arg), final_shape)
    return (lowered, [])