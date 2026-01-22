import osqp
import numpy as np
import numpy.random as npr
from scipy import sparse
from scipy.optimize import approx_fprime
import numpy.testing as npt
import unittest
def test_dl_dq(self, verbose=False):
    n, m = (5, 5)
    prob = self.get_prob(n=n, m=m, P_scale=100.0, A_scale=100.0)
    P, q, A, l, u, true_x = prob

    def grad(q):
        [dP, dq, dA, dl, du] = self.get_grads(P, q, A, l, u, true_x)
        return dq

    def f(q):
        m = osqp.OSQP()
        m.setup(P, q, A, l, u, eps_abs=eps_abs, eps_rel=eps_rel, max_iter=max_iter, verbose=False)
        res = m.solve()
        if res.info.status != 'solved':
            raise ValueError('Problem not solved!')
        x_hat = res.x
        return 0.5 * np.sum(np.square(x_hat - true_x))
    dq = grad(q)
    dq_fd = approx_fprime(q, f, grad_precision)
    if verbose:
        print('dq_fd: ', np.round(dq_fd, decimals=4))
        print('dq: ', np.round(dq, decimals=4))
    npt.assert_allclose(dq_fd, dq, rtol=rel_tol, atol=abs_tol)