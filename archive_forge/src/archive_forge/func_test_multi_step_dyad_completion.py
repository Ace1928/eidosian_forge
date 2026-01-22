import numpy as np
import cvxpy as cp
from cvxpy.tests.base_test import BaseTest
def test_multi_step_dyad_completion(self) -> None:
    """
        Consider four market equilibrium problems.

        The budgets "b" in these problems are chosen so that canonicalization
        of geo_mean(u, b) hits a recursive code-path in power_tools.dyad_completion(...).

        The reference solution is computed by taking the log of the geo_mean objective,
        which has the effect of making the problem ExpCone representable.
        """
    if 'MOSEK' in cp.installed_solvers():
        log_solve_args = {'solver': 'MOSEK'}
    else:
        log_solve_args = {'solver': 'ECOS'}
    n_buyer = 5
    n_items = 7
    np.random.seed(0)
    V = 0.5 * (1 + np.random.rand(n_buyer, n_items))
    X = cp.Variable(shape=(n_buyer, n_items), nonneg=True)
    cons = [cp.sum(X, axis=0) <= 1]
    u = cp.sum(cp.multiply(V, X), axis=1)
    bs = np.array([[110, 14, 6, 77, 108], [15.0, 4.0, 8.0, 0.0, 9.0], [14.0, 21.0, 217.0, 57.0, 6.0], [3.0, 36.0, 77.0, 8.0, 8.0]])
    for i, b in enumerate(bs):
        log_objective = cp.Maximize(b @ cp.log(u))
        log_prob = cp.Problem(log_objective, cons)
        log_prob.solve(**log_solve_args)
        expect_X = X.value
        geo_objective = cp.Maximize(cp.geo_mean(u, b))
        geo_prob = cp.Problem(geo_objective, cons)
        geo_prob.solve()
        actual_X = X.value
        try:
            self.assertItemsAlmostEqual(actual_X, expect_X, places=3)
        except AssertionError as e:
            print(f'Failure at index {i} (when b={str(b)}).')
            log_prob.solve(**log_solve_args, verbose=True)
            print(X.value)
            geo_prob.solve(verbose=True)
            print(X.value)
            print('The valuation matrix was')
            print(V)
            raise e