import time
import numpy as np
from scipy.sparse.linalg import LinearOperator
from .._differentiable_functions import VectorFunction
from .._constraints import (
from .._hessian_update_strategy import BFGS
from .._optimize import OptimizeResult
from .._differentiable_functions import ScalarFunction
from .equality_constrained_sqp import equality_constrained_sqp
from .canonical_constraint import (CanonicalConstraint,
from .tr_interior_point import tr_interior_point
from .report import BasicReport, SQPReport, IPReport
def update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info):
    state.nit += 1
    state.nfev = objective.nfev
    state.njev = objective.ngev
    state.nhev = objective.nhev
    state.constr_nfev = [c.fun.nfev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    state.constr_njev = [c.fun.njev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    state.constr_nhev = [c.fun.nhev if isinstance(c.fun, VectorFunction) else 0 for c in prepared_constraints]
    if not last_iteration_failed:
        state.x = x
        state.fun = objective.f
        state.grad = objective.g
        state.v = [c.fun.v for c in prepared_constraints]
        state.constr = [c.fun.f for c in prepared_constraints]
        state.jac = [c.fun.J for c in prepared_constraints]
        state.lagrangian_grad = np.copy(state.grad)
        for c in prepared_constraints:
            state.lagrangian_grad += c.fun.J.T.dot(c.fun.v)
        state.optimality = np.linalg.norm(state.lagrangian_grad, np.inf)
        state.constr_violation = 0
        for i in range(len(prepared_constraints)):
            lb, ub = prepared_constraints[i].bounds
            c = state.constr[i]
            state.constr_violation = np.max([state.constr_violation, np.max(lb - c), np.max(c - ub)])
    state.execution_time = time.time() - start_time
    state.tr_radius = tr_radius
    state.constr_penalty = constr_penalty
    state.cg_niter += cg_info['niter']
    state.cg_stop_cond = cg_info['stop_cond']
    return state