from typing import List, Tuple
import numpy as np
import cvxpy as cp
from cvxpy.atoms.affine.upper_tri import upper_tri
from cvxpy.constraints.constraint import Constraint
from cvxpy.constraints.exponential import (
from cvxpy.constraints.zero import Zero
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.dcp2cone.canonicalizers.von_neumann_entr_canon import (
def von_neumann_entr_canon_dispatch(expr, args):
    if expr.quad_approx:
        return von_neumann_entr_QuadApprox(expr, args)
    else:
        return von_neumann_entr_canon(expr, args)