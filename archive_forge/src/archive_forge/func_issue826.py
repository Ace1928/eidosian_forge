import sys
import unittest
import numpy as np
import cvxpy as cp
def issue826() -> None:
    n = 2 ** 8
    m = int(2 ** 32 / n) + 1
    vals = np.arange(m * n, dtype=np.double) / 1000.0
    A = vals.reshape(n, m)
    x = cp.Variable(shape=(m,))
    cons = [A @ x >= 0]
    prob = cp.Problem(cp.Maximize(0), cons)
    data = prob.get_problem_data(solver='SCS')
    vals_canon = data[0]['A'].data
    tester = unittest.TestCase()
    diff = vals - vals_canon
    err = np.abs(diff)
    tester.assertLessEqual(err, 0.001)
    print('\t issue826 test finished')