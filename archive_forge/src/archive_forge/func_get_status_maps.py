import numpy as np
import cvxpy.settings as s
from cvxpy.constraints import SOC
from cvxpy.reductions.solution import Solution
from cvxpy.reductions.solvers import utilities
from cvxpy.reductions.solvers.conic_solvers.conic_solver import (
def get_status_maps():
    """Create status maps from Xpress to CVXPY
    """
    import xpress as xp
    status_map_lp = {xp.lp_unstarted: s.SOLVER_ERROR, xp.lp_optimal: s.OPTIMAL, xp.lp_infeas: s.INFEASIBLE, xp.lp_cutoff: s.OPTIMAL_INACCURATE, xp.lp_unfinished: s.OPTIMAL_INACCURATE, xp.lp_unbounded: s.UNBOUNDED, xp.lp_cutoff_in_dual: s.OPTIMAL_INACCURATE, xp.lp_unsolved: s.OPTIMAL_INACCURATE, xp.lp_nonconvex: s.SOLVER_ERROR}
    status_map_mip = {xp.mip_not_loaded: s.SOLVER_ERROR, xp.mip_lp_not_optimal: s.SOLVER_ERROR, xp.mip_lp_optimal: s.SOLVER_ERROR, xp.mip_no_sol_found: s.SOLVER_ERROR, xp.mip_solution: s.OPTIMAL_INACCURATE, xp.mip_infeas: s.INFEASIBLE, xp.mip_optimal: s.OPTIMAL, xp.mip_unbounded: s.UNBOUNDED}
    return (status_map_lp, status_map_mip)