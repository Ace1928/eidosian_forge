import scipy.sparse as sps
import numpy as np
from .equality_constrained_sqp import equality_constrained_sqp
from scipy.sparse.linalg import LinearOperator
def tr_interior_point(fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, x0, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0, stop_criteria, enforce_feasibility, xtol, state, initial_barrier_parameter, initial_tolerance, initial_penalty, initial_trust_radius, factorization_method):
    """Trust-region interior points method.

    Solve problem:
        minimize fun(x)
        subject to: constr_ineq(x) <= 0
                    constr_eq(x) = 0
    using trust-region interior point method described in [1]_.
    """
    BOUNDARY_PARAMETER = 0.995
    BARRIER_DECAY_RATIO = 0.2
    TRUST_ENLARGEMENT = 5
    if enforce_feasibility is None:
        enforce_feasibility = np.zeros(n_ineq, bool)
    barrier_parameter = initial_barrier_parameter
    tolerance = initial_tolerance
    trust_radius = initial_trust_radius
    s0 = np.maximum(-1.5 * constr_ineq0, np.ones(n_ineq))
    subprob = BarrierSubproblem(x0, s0, fun, grad, lagr_hess, n_vars, n_ineq, n_eq, constr, jac, barrier_parameter, tolerance, enforce_feasibility, stop_criteria, xtol, fun0, grad0, constr_ineq0, jac_ineq0, constr_eq0, jac_eq0)
    z = np.hstack((x0, s0))
    fun0_subprob, constr0_subprob = (subprob.fun0, subprob.constr0)
    grad0_subprob, jac0_subprob = (subprob.grad0, subprob.jac0)
    trust_lb = np.hstack((np.full(subprob.n_vars, -np.inf), np.full(subprob.n_ineq, -BOUNDARY_PARAMETER)))
    trust_ub = np.full(subprob.n_vars + subprob.n_ineq, np.inf)
    while True:
        z, state = equality_constrained_sqp(subprob.function_and_constraints, subprob.gradient_and_jacobian, subprob.lagrangian_hessian, z, fun0_subprob, grad0_subprob, constr0_subprob, jac0_subprob, subprob.stop_criteria, state, initial_penalty, trust_radius, factorization_method, trust_lb, trust_ub, subprob.scaling)
        if subprob.terminate:
            break
        trust_radius = max(initial_trust_radius, TRUST_ENLARGEMENT * state.tr_radius)
        barrier_parameter *= BARRIER_DECAY_RATIO
        tolerance *= BARRIER_DECAY_RATIO
        subprob.update(barrier_parameter, tolerance)
        fun0_subprob, constr0_subprob = subprob.function_and_constraints(z)
        grad0_subprob, jac0_subprob = subprob.gradient_and_jacobian(z)
    x = subprob.get_variables(z)
    return (x, state)