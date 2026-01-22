import osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt
import unittest
def test_dl_dP(self, verbose=False):
    n, m = (3, 3)
    prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
    P, q, A, l, u, true_x = prob
    P_idx = P.nonzero()

    def grad(P_val):
        P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
        [dP, dq, dA, dl, du] = self.get_grads(P_qp, q, A, l, u, true_x)
        return dP

    def f(P_val):
        P_qp = sparse.csc_matrix((P_val, P_idx), shape=P.shape)
        m = osqp.OSQP()
        m.setup(P_qp, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, verbose=False)
        res = m.solve()
        if res.info.status != 'solved':
            raise ValueError('Problem not solved!')
        x_hat = res.x
        return 0.5 * np.sum(np.square(x_hat - true_x))
    dP = grad(P.data)
    dP_fd_val = approx_fprime(P.data, f, grad_precision)
    dP_fd = sparse.csc_matrix((dP_fd_val, P_idx), shape=P.shape)
    dP_fd = (dP_fd + dP_fd.T) / 2
    if verbose:
        print('dP_fd: ', np.round(dP_fd.data, decimals=4))
        print('dA: ', np.round(dP.data, decimals=4))
    npt.assert_allclose(dP.todense(), dP_fd.todense(), rtol=rel_tol, atol=abs_tol)