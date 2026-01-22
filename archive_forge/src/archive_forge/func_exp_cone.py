import numpy as np
from cvxpy import Maximize, Minimize, Problem
from cvxpy.atoms import diag, exp, hstack, pnorm
from cvxpy.constraints import SOC, ExpCone, NonNeg
from cvxpy.error import SolverError
from cvxpy.expressions.constants import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.reductions.cvx_attr2constr import CvxAttr2Constr
from cvxpy.reductions.dcp2cone.cone_matrix_stuffing import ConeMatrixStuffing
from cvxpy.reductions.flip_objective import FlipObjective
from cvxpy.reductions.solvers.conic_solvers.ecos_conif import ECOS
from cvxpy.tests.base_test import BaseTest
from cvxpy.tests.solver_test_helpers import SolverTestHelper
def exp_cone(self) -> None:
    """Test exponential cone problems.
        """
    for solver in self.solvers:
        p = Problem(Minimize(self.b), [exp(self.a) <= self.b, self.a >= 1])
        pmod = Problem(Minimize(self.b), [ExpCone(self.a, Constant(1), self.b), self.a >= 1])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        p_new = ConeMatrixStuffing().apply(pmod)
        if not solver.accepts(p_new[0]):
            return
        result = p.solve(solver.name())
        sltn = solve_wrapper(solver, p_new[0])
        self.assertAlmostEqual(sltn.opt_val, result, places=1)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result, places=1)
        for var in pmod.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id], var.value, places=1)
        p = Problem(Minimize(self.b), [exp(self.a / 2 + self.c) <= self.b + 5, self.a >= 1, self.c >= 5])
        pmod = Problem(Minimize(self.b), [ExpCone(self.a / 2 + self.c, Constant(1), self.b + 5), self.a >= 1, self.c >= 5])
        self.assertTrue(ConeMatrixStuffing().accepts(pmod))
        result = p.solve(solver.name())
        p_new = ConeMatrixStuffing().apply(pmod)
        sltn = solve_wrapper(solver, p_new[0])
        self.assertAlmostEqual(sltn.opt_val, result, places=0)
        inv_sltn = ConeMatrixStuffing().invert(sltn, p_new[1])
        self.assertAlmostEqual(inv_sltn.opt_val, result, places=0)
        for var in pmod.variables():
            self.assertItemsAlmostEqual(inv_sltn.primal_vars[var.id], var.value, places=0)