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
def update_state_ip(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info, barrier_parameter, barrier_tolerance):
    state = update_state_sqp(state, x, last_iteration_failed, objective, prepared_constraints, start_time, tr_radius, constr_penalty, cg_info)
    state.barrier_parameter = barrier_parameter
    state.barrier_tolerance = barrier_tolerance
    return state