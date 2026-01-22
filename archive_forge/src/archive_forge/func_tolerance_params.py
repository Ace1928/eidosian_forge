from __future__ import annotations
import warnings
from collections import defaultdict
import numpy as np
import scipy as sp
import cvxpy.settings as s
from cvxpy.constraints import PSD, SOC, ExpCone, PowCone3D
from cvxpy.reductions.cone2cone import affine2direct as a2d
from cvxpy.reductions.cone2cone.affine2direct import Dualize, Slacks
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers.conic_solvers.conic_solver import ConicSolver
from cvxpy.reductions.solvers.utilities import expcone_permutor
@staticmethod
def tolerance_params() -> tuple[str]:
    return ('MSK_DPAR_INTPNT_CO_TOL_DFEAS', 'MSK_DPAR_INTPNT_CO_TOL_INFEAS', 'MSK_DPAR_INTPNT_CO_TOL_MU_RED', 'MSK_DPAR_INTPNT_CO_TOL_PFEAS', 'MSK_DPAR_INTPNT_CO_TOL_REL_GAP', 'MSK_DPAR_INTPNT_TOL_DFEAS', 'MSK_DPAR_INTPNT_TOL_INFEAS', 'MSK_DPAR_INTPNT_TOL_MU_RED', 'MSK_DPAR_INTPNT_TOL_PFEAS', 'MSK_DPAR_INTPNT_TOL_REL_GAP', 'MSK_DPAR_BASIS_REL_TOL_S', 'MSK_DPAR_BASIS_TOL_S', 'MSK_DPAR_BASIS_TOL_X', 'MSK_DPAR_MIO_TOL_ABS_GAP', 'MSK_DPAR_MIO_TOL_ABS_RELAX_INT', 'MSK_DPAR_MIO_TOL_FEAS', 'MSK_DPAR_MIO_TOL_REL_GAP')