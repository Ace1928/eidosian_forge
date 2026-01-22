import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def test_gp_env(self):
    with gp.Env(params=self.env_options) as env:
        prob = generate_lp()
        solver = GUROBI(msg=False, env=env, **self.options)
        prob.solve(solver)
        solver.close()
    check_dummy_env()