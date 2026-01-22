import numpy as np
from scipy.optimize._trustregion_exact import (
from scipy.linalg import (svd, get_lapack_funcs, det, qr, norm)
from numpy.testing import (assert_array_equal,
def random_entry(n, min_eig, max_eig, case):
    rand = np.random.uniform(-1, 1, (n, n))
    Q, _, _ = qr(rand, pivoting='True')
    eigvalues = np.random.uniform(min_eig, max_eig, n)
    eigvalues = np.sort(eigvalues)[::-1]
    Qaux = np.multiply(eigvalues, Q)
    A = np.dot(Qaux, Q.T)
    if case == 'hard':
        g = np.zeros(n)
        g[:-1] = np.random.uniform(-1, 1, n - 1)
        g = np.dot(Q, g)
    elif case == 'jac_equal_zero':
        g = np.zeros(n)
    else:
        g = np.random.uniform(-1, 1, n)
    return (A, g)