from typing import List, Tuple
import cvxpy.lin_ops.lin_op as lo
from cvxpy.atoms.affine.affine_atom import AffAtom
from cvxpy.constraints.constraint import Constraint
def validate_real_square(arg):
    ndim_test = len(arg.shape) == 2
    if not ndim_test:
        raise ValueError('The input must be a square matrix.')
    elif arg.shape[0] != arg.shape[1]:
        raise ValueError('The input must be a square matrix.')
    elif not arg.is_real():
        raise ValueError('The input must be a real matrix.')