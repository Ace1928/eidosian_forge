from collections import defaultdict
from typing import Tuple
import numpy as np
import scipy.sparse as sp
from cvxpy.atoms.affine.reshape import reshape
from cvxpy.atoms.affine.vec import vec
from cvxpy.constraints.nonpos import NonNeg, NonPos
from cvxpy.constraints.zero import Zero
from cvxpy.cvxcore.python import canonInterface
def group_constraints(constraints):
    """Organize the constraints into a dictionary keyed by constraint names.

    Parameters
    ---------
    constraints : list of constraints

    Returns
    -------
    dict
        A dict keyed by constraint types where dict[cone_type] maps to a list
        of exactly those constraints that are of type cone_type.
    """
    constr_map = defaultdict(list)
    for c in constraints:
        constr_map[type(c)].append(c)
    return constr_map