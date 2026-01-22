import numpy as np
import scipy.sparse as sps
def initial_constraints_as_canonical(n, prepared_constraints, sparse_jacobian):
    """Convert initial values of the constraints to the canonical format.

    The purpose to avoid one additional call to the constraints at the initial
    point. It takes saved values in `PreparedConstraint`, modififies and
    concatenates them to the canonical constraint format.
    """
    c_eq = []
    c_ineq = []
    J_eq = []
    J_ineq = []
    for c in prepared_constraints:
        f = c.fun.f
        J = c.fun.J
        lb, ub = c.bounds
        if np.all(lb == ub):
            c_eq.append(f - lb)
            J_eq.append(J)
        elif np.all(lb == -np.inf):
            finite_ub = ub < np.inf
            c_ineq.append(f[finite_ub] - ub[finite_ub])
            J_ineq.append(J[finite_ub])
        elif np.all(ub == np.inf):
            finite_lb = lb > -np.inf
            c_ineq.append(lb[finite_lb] - f[finite_lb])
            J_ineq.append(-J[finite_lb])
        else:
            lb_inf = lb == -np.inf
            ub_inf = ub == np.inf
            equal = lb == ub
            less = lb_inf & ~ub_inf
            greater = ub_inf & ~lb_inf
            interval = ~equal & ~lb_inf & ~ub_inf
            c_eq.append(f[equal] - lb[equal])
            c_ineq.append(f[less] - ub[less])
            c_ineq.append(lb[greater] - f[greater])
            c_ineq.append(f[interval] - ub[interval])
            c_ineq.append(lb[interval] - f[interval])
            J_eq.append(J[equal])
            J_ineq.append(J[less])
            J_ineq.append(-J[greater])
            J_ineq.append(J[interval])
            J_ineq.append(-J[interval])
    c_eq = np.hstack(c_eq) if c_eq else np.empty(0)
    c_ineq = np.hstack(c_ineq) if c_ineq else np.empty(0)
    if sparse_jacobian:
        vstack = sps.vstack
        empty = sps.csr_matrix((0, n))
    else:
        vstack = np.vstack
        empty = np.empty((0, n))
    J_eq = vstack(J_eq) if J_eq else empty
    J_ineq = vstack(J_ineq) if J_ineq else empty
    return (c_eq, c_ineq, J_eq, J_ineq)