from __future__ import annotations
import warnings
import numpy as np
from cvxpy.atoms import EXP_ATOMS, NONPOS_ATOMS, PSD_ATOMS, SOC_ATOMS
from cvxpy.constraints import (
from cvxpy.constraints.exponential import OpRelEntrConeQuad, RelEntrConeQuad
from cvxpy.error import DCPError, DGPError, DPPError, SolverError
from cvxpy.problems.objective import Maximize
from cvxpy.reductions.chain import Chain
from cvxpy.reductions.complex2real import complex2real
from cvxpy.reductions.cone2cone.approximations import APPROX_CONES, QuadApprox
from cvxpy.reductions.cone2cone.exotic2common import (
from cvxpy.reductions.cone2cone.soc2psd import SOC2PSD
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.dcp2cone.dcp2cone import Dcp2Cone
from cvxpy.reductions.dgp2dcp.dgp2dcp import Dgp2Dcp
from cvxpy.reductions.discrete2mixedint.valinvec2mixedint import (
from cvxpy.reductions.eval_params import EvalParams
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.qp2quad_form import qp2symbolic_qp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing
from cvxpy.reductions.reduction import Reduction
from cvxpy.reductions.solvers import defines as slv_def
from cvxpy.reductions.solvers.constant_solver import ConstantSolver
from cvxpy.reductions.solvers.solver import Solver
from cvxpy.settings import ECOS, PARAM_THRESHOLD
from cvxpy.utilities.debug_tools import build_non_disciplined_error_msg
def construct_solving_chain(problem, candidates, gp: bool=False, enforce_dpp: bool=False, ignore_dpp: bool=False, canon_backend: str | None=None, solver_opts: dict | None=None, specified_solver: str | None=None) -> 'SolvingChain':
    """Build a reduction chain from a problem to an installed solver.

    Note that if the supplied problem has 0 variables, then the solver
    parameter will be ignored.

    Parameters
    ----------
    problem : Problem
        The problem for which to build a chain.
    candidates : dict
        Dictionary of candidate solvers divided in qp_solvers
        and conic_solvers.
    gp : bool
        If True, the problem is parsed as a Disciplined Geometric Program
        instead of as a Disciplined Convex Program.
    enforce_dpp : bool, optional
        When True, a DPPError will be thrown when trying to parse a non-DPP
        problem (instead of just a warning). Defaults to False.
    ignore_dpp : bool, optional
        When True, DPP problems will be treated as non-DPP,
        which may speed up compilation. Defaults to False.
    canon_backend : str, optional
        'CPP' (default) | 'SCIPY'
        Specifies which backend to use for canonicalization, which can affect
        compilation time. Defaults to None, i.e., selecting the default
        backend.
    solver_opts : dict, optional
        Additional arguments to pass to the solver.
    specified_solver: str, optional
        A solver specified by the user.

    Returns
    -------
    SolvingChain
        A SolvingChain that can be used to solve the problem.

    Raises
    ------
    SolverError
        Raised if no suitable solver exists among the installed solvers, or
        if the target solver is not installed.
    """
    if len(problem.variables()) == 0:
        return SolvingChain(reductions=[ConstantSolver()])
    reductions = _reductions_for_problem_class(problem, candidates, gp, solver_opts)
    dpp_context = 'dcp' if not gp else 'dgp'
    if ignore_dpp or not problem.is_dpp(dpp_context):
        if ignore_dpp:
            reductions = [EvalParams()] + reductions
        elif not enforce_dpp:
            warnings.warn(DPP_ERROR_MSG)
            reductions = [EvalParams()] + reductions
        else:
            raise DPPError(DPP_ERROR_MSG)
    elif any((param.is_complex() for param in problem.parameters())):
        reductions = [EvalParams()] + reductions
    else:
        n_parameters = sum((np.prod(param.shape) for param in problem.parameters()))
        if n_parameters >= PARAM_THRESHOLD:
            warnings.warn("Your problem has too many parameters for efficient DPP compilation. We suggest setting 'ignore_dpp = True'.")
    use_quad = True if solver_opts is None else solver_opts.get('use_quad_obj', True)
    if _solve_as_qp(problem, candidates) and use_quad:
        solver = candidates['qp_solvers'][0]
        solver_instance = slv_def.SOLVER_MAP_QP[solver]
        reductions += [QpMatrixStuffing(canon_backend=canon_backend), solver_instance]
        return SolvingChain(reductions=reductions)
    if not candidates['conic_solvers']:
        raise SolverError('Problem could not be reduced to a QP, and no conic solvers exist among candidate solvers (%s).' % candidates)
    constr_types = set()
    for c in problem.constraints:
        constr_types.add(type(c))
    ex_cos = [ct for ct in constr_types if ct in EXOTIC_CONES]
    approx_cos = [ct for ct in constr_types if ct in APPROX_CONES]
    for co in ex_cos:
        sim_cos = EXOTIC_CONES[co]
        constr_types.update(sim_cos)
        constr_types.remove(co)
    for co in approx_cos:
        app_cos = APPROX_CONES[co]
        constr_types.update(app_cos)
        constr_types.remove(co)
    cones = []
    atoms = problem.atoms()
    if SOC in constr_types or any((atom in SOC_ATOMS for atom in atoms)):
        cones.append(SOC)
    if ExpCone in constr_types or any((atom in EXP_ATOMS for atom in atoms)):
        cones.append(ExpCone)
    if any((t in constr_types for t in [Inequality, NonPos, NonNeg])) or any((atom in NONPOS_ATOMS for atom in atoms)):
        cones.append(NonNeg)
    if Equality in constr_types or Zero in constr_types:
        cones.append(Zero)
    if PSD in constr_types or any((atom in PSD_ATOMS for atom in atoms)) or any((v.is_psd() or v.is_nsd() for v in problem.variables())):
        cones.append(PSD)
    if PowCone3D in constr_types:
        cones.append(PowCone3D)
    has_constr = len(cones) > 0 or len(problem.constraints) > 0
    for solver in candidates['conic_solvers']:
        solver_instance = slv_def.SOLVER_MAP_CONIC[solver]
        if problem.is_mixed_integer():
            supported_constraints = solver_instance.MI_SUPPORTED_CONSTRAINTS
        else:
            supported_constraints = solver_instance.SUPPORTED_CONSTRAINTS
        unsupported_constraints = [cone for cone in cones if cone not in supported_constraints]
        if has_constr or not solver_instance.REQUIRES_CONSTR:
            if ex_cos:
                reductions.append(Exotic2Common())
            if RelEntrConeQuad in approx_cos or OpRelEntrConeQuad in approx_cos:
                reductions.append(QuadApprox())
            if solver_opts is None:
                use_quad_obj = True
            else:
                use_quad_obj = solver_opts.get('use_quad_obj', True)
            quad_obj = use_quad_obj and solver_instance.supports_quad_obj() and problem.objective.expr.has_quadratic_term()
            reductions += [Dcp2Cone(quad_obj=quad_obj), CvxAttr2Constr()]
            if all((c in supported_constraints for c in cones)):
                if solver == ECOS and specified_solver is None:
                    warnings.warn(ECOS_DEPRECATION_MSG, FutureWarning)
                reductions += [ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend), solver_instance]
                return SolvingChain(reductions=reductions)
            elif all((c == SOC for c in unsupported_constraints)) and PSD in supported_constraints:
                reductions += [SOC2PSD(), ConeMatrixStuffing(quad_obj=quad_obj, canon_backend=canon_backend), solver_instance]
                return SolvingChain(reductions=reductions)
    raise SolverError('Either candidate conic solvers (%s) do not support the cones output by the problem (%s), or there are not enough constraints in the problem.' % (candidates['conic_solvers'], ', '.join([cone.__name__ for cone in cones])))