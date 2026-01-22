import numpy as np
import scipy.sparse as sp
import cvxpy.settings as s
from cvxpy.constraints import NonNeg, Zero
from cvxpy.reductions.cvx_attr2constr import convex_attributes
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import (
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.reductions.utilities import group_constraints

        Construct QP problem data stored in a dictionary.
        The QP has the following form

            minimize      1/2 x' P x + q' x
            subject to    A x =  b
                          F x <= g

        