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
def parse_eps_keyword(solver_opts: dict) -> dict:
    """
        Parse the eps keyword and update the corresponding MOSEK parameters.
        If additional tolerances are specified explicitly, they take precedence over the
        eps keyword.
        """
    if 'eps' not in solver_opts:
        return solver_opts
    tol_params = MOSEK.tolerance_params()
    mosek_params = solver_opts.get('mosek_params', dict())
    assert not any((MOSEK.is_param(p) for p in mosek_params)), 'The eps keyword is not compatible with (deprecated) Mosek enum parameters.             Use the string parameters instead.'
    solver_opts['mosek_params'] = mosek_params
    eps = solver_opts.pop('eps')
    for tol_param in tol_params:
        solver_opts['mosek_params'][tol_param] = solver_opts['mosek_params'].get(tol_param, eps)
    return solver_opts