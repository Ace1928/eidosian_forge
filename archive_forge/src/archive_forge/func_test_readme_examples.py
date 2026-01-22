from __future__ import print_function
import unittest
import numpy as np
import cvxpy as cvx
import cvxpy.interface as intf
from cvxpy.reductions.solvers.conic_solvers import ecos_conif
from cvxpy.tests.base_test import BaseTest
def test_readme_examples(self) -> None:
    import numpy
    numpy.random.seed(1)
    m = 30
    n = 20
    A = numpy.random.randn(m, n)
    b = numpy.random.randn(m)
    x = cvx.Variable(n)
    objective = cvx.Minimize(cvx.sum_squares(A @ x - b))
    constraints = [0 <= x, x <= 1]
    p = cvx.Problem(objective, constraints)
    p.solve(solver=cvx.SCS)
    print(x.value)
    print(constraints[0].dual_value)
    a = cvx.Variable()
    x = cvx.Variable(5)
    A = cvx.Variable((4, 7))
    m = cvx.Parameter(nonneg=True)
    cvx.Parameter(5)
    G = cvx.Parameter((4, 7), nonpos=True)
    G.value = -numpy.ones((4, 7))
    with self.assertRaises(Exception) as cm:
        G.value = numpy.ones((4, 7))
    self.assertEqual(str(cm.exception), 'Parameter value must be nonpositive.')
    a = cvx.Variable()
    x = cvx.Variable(5)
    expr = 2 * x
    expr = expr - a
    expr = cvx.sum(expr) + cvx.norm(x, 2)
    import numpy as np
    n = 10
    m = 5
    A = np.random.randn(n, m)
    b = np.random.randn(n)
    gamma = cvx.Parameter(nonneg=True)
    x = cvx.Variable(m)
    objective = cvx.Minimize(cvx.sum_squares(A @ x - b) + gamma * cvx.norm(x, 1))
    p = cvx.Problem(objective)

    def get_x(gamma_value):
        gamma.value = gamma_value
        p.solve(solver=cvx.SCS)
        return x.value
    gammas = np.logspace(-1, 2, num=2)
    [get_x(value) for value in gammas]
    n = 10
    mu = np.random.randn(1, n)
    sigma = np.random.randn(n, n)
    sigma = sigma.T.dot(sigma)
    gamma = cvx.Parameter(nonneg=True)
    gamma.value = 1
    x = cvx.Variable(n)
    expected_return = mu @ x
    risk = cvx.quad_form(x, sigma)
    objective = cvx.Maximize(expected_return - gamma * risk)
    p = cvx.Problem(objective, [cvx.sum(x) == 1])
    p.solve(solver=cvx.SCS)
    print(expected_return.value)
    print(risk.value)
    N = 50
    M = 40
    n = 10
    data = []
    for i in range(N):
        data += [(1, np.random.normal(loc=1.0, scale=2.0, size=n))]
    for i in range(M):
        data += [(-1, np.random.normal(loc=-1.0, scale=2.0, size=n))]
    gamma = cvx.Parameter(nonneg=True)
    gamma.value = 0.1
    a = cvx.Variable(n)
    b = cvx.Variable()
    slack = [cvx.pos(1 - label * (sample.T @ a - b)) for label, sample in data]
    objective = cvx.Minimize(cvx.norm(a, 2) + gamma * sum(slack))
    p = cvx.Problem(objective)
    p.solve(solver=cvx.SCS)
    errors = 0
    for label, sample in data:
        if label * (sample.T @ a - b).value < 0:
            errors += 1
    print('%s misclassifications' % errors)
    print(a.value)
    print(b.value)