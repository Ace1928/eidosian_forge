from scipy.sparse import (linalg, bmat, csc_matrix)
from math import copysign
import numpy as np
from numpy.linalg import norm
def projected_cg(H, c, Z, Y, b, trust_radius=np.inf, lb=None, ub=None, tol=None, max_iter=None, max_infeasible_iter=None, return_all=False):
    """Solve EQP problem with projected CG method.

    Solve equality-constrained quadratic programming problem
    ``min 1/2 x.T H x + x.t c``  subject to ``A x + b = 0`` and,
    possibly, to trust region constraints ``||x|| < trust_radius``
    and box constraints ``lb <= x <= ub``.

    Parameters
    ----------
    H : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for computing ``H v``.
    c : array_like, shape (n,)
        Gradient of the quadratic objective function.
    Z : LinearOperator (or sparse matrix or ndarray), shape (n, n)
        Operator for projecting ``x`` into the null space of A.
    Y : LinearOperator,  sparse matrix, ndarray, shape (n, m)
        Operator that, for a given a vector ``b``, compute smallest
        norm solution of ``A x + b = 0``.
    b : array_like, shape (m,)
        Right-hand side of the constraint equation.
    trust_radius : float, optional
        Trust radius to be considered. By default, uses ``trust_radius=inf``,
        which means no trust radius at all.
    lb : array_like, shape (n,), optional
        Lower bounds to each one of the components of ``x``.
        If ``lb[i] = -Inf`` the lower bound for the i-th
        component is just ignored (default).
    ub : array_like, shape (n, ), optional
        Upper bounds to each one of the components of ``x``.
        If ``ub[i] = Inf`` the upper bound for the i-th
        component is just ignored (default).
    tol : float, optional
        Tolerance used to interrupt the algorithm.
    max_iter : int, optional
        Maximum algorithm iterations. Where ``max_inter <= n-m``.
        By default, uses ``max_iter = n-m``.
    max_infeasible_iter : int, optional
        Maximum infeasible (regarding box constraints) iterations the
        algorithm is allowed to take.
        By default, uses ``max_infeasible_iter = n-m``.
    return_all : bool, optional
        When ``true``, return the list of all vectors through the iterations.

    Returns
    -------
    x : array_like, shape (n,)
        Solution of the EQP problem.
    info : Dict
        Dictionary containing the following:

            - niter : Number of iterations.
            - stop_cond : Reason for algorithm termination:
                1. Iteration limit was reached;
                2. Reached the trust-region boundary;
                3. Negative curvature detected;
                4. Tolerance was satisfied.
            - allvecs : List containing all intermediary vectors (optional).
            - hits_boundary : True if the proposed step is on the boundary
              of the trust region.

    Notes
    -----
    Implementation of Algorithm 6.2 on [1]_.

    In the absence of spherical and box constraints, for sufficient
    iterations, the method returns a truly optimal result.
    In the presence of those constraints, the value returned is only
    a inexpensive approximation of the optimal value.

    References
    ----------
    .. [1] Gould, Nicholas IM, Mary E. Hribar, and Jorge Nocedal.
           "On the solution of equality constrained quadratic
            programming problems arising in optimization."
            SIAM Journal on Scientific Computing 23.4 (2001): 1376-1395.
    """
    CLOSE_TO_ZERO = 1e-25
    n, = np.shape(c)
    m, = np.shape(b)
    x = Y.dot(-b)
    r = Z.dot(H.dot(x) + c)
    g = Z.dot(r)
    p = -g
    if return_all:
        allvecs = [x]
    H_p = H.dot(p)
    rt_g = norm(g) ** 2
    tr_distance = trust_radius - norm(x)
    if tr_distance < 0:
        raise ValueError('Trust region problem does not have a solution.')
    elif tr_distance < CLOSE_TO_ZERO:
        info = {'niter': 0, 'stop_cond': 2, 'hits_boundary': True}
        if return_all:
            allvecs.append(x)
            info['allvecs'] = allvecs
        return (x, info)
    if tol is None:
        tol = max(min(0.01 * np.sqrt(rt_g), 0.1 * rt_g), CLOSE_TO_ZERO)
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)
    if max_iter is None:
        max_iter = n - m
    max_iter = min(max_iter, n - m)
    if max_infeasible_iter is None:
        max_infeasible_iter = n - m
    hits_boundary = False
    stop_cond = 1
    counter = 0
    last_feasible_x = np.zeros_like(x)
    k = 0
    for i in range(max_iter):
        if rt_g < tol:
            stop_cond = 4
            break
        k += 1
        pt_H_p = H_p.dot(p)
        if pt_H_p <= 0:
            if np.isinf(trust_radius):
                raise ValueError('Negative curvature not allowed for unrestricted problems.')
            else:
                _, alpha, intersect = box_sphere_intersections(x, p, lb, ub, trust_radius, entire_line=True)
                if intersect:
                    x = x + alpha * p
                x = reinforce_box_boundaries(x, lb, ub)
                stop_cond = 3
                hits_boundary = True
                break
        alpha = rt_g / pt_H_p
        x_next = x + alpha * p
        if np.linalg.norm(x_next) >= trust_radius:
            _, theta, intersect = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                x = x + theta * alpha * p
            x = reinforce_box_boundaries(x, lb, ub)
            stop_cond = 2
            hits_boundary = True
            break
        if inside_box_boundaries(x_next, lb, ub):
            counter = 0
        else:
            counter += 1
        if counter > 0:
            _, theta, intersect = box_sphere_intersections(x, alpha * p, lb, ub, trust_radius)
            if intersect:
                last_feasible_x = x + theta * alpha * p
                last_feasible_x = reinforce_box_boundaries(last_feasible_x, lb, ub)
                counter = 0
        if counter > max_infeasible_iter:
            break
        if return_all:
            allvecs.append(x_next)
        r_next = r + alpha * H_p
        g_next = Z.dot(r_next)
        rt_g_next = norm(g_next) ** 2
        beta = rt_g_next / rt_g
        p = -g_next + beta * p
        x = x_next
        g = g_next
        r = g_next
        rt_g = norm(g) ** 2
        H_p = H.dot(p)
    if not inside_box_boundaries(x, lb, ub):
        x = last_feasible_x
        hits_boundary = True
    info = {'niter': k, 'stop_cond': stop_cond, 'hits_boundary': hits_boundary}
    if return_all:
        info['allvecs'] = allvecs
    return (x, info)