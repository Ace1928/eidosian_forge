import unittest
from pulp import GUROBI, LpProblem, LpVariable, const
def test_multiple_gp_env(self):
    with gp.Env(params=self.env_options) as env:
        solver = GUROBI(msg=False, env=env)
        prob = generate_lp()
        prob.solve(solver)
        solver.close()
        solver2 = GUROBI(msg=False, env=env)
        prob2 = generate_lp()
        prob2.solve(solver2)
        solver2.close()
    check_dummy_env()