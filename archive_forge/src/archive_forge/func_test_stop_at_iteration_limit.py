import pyomo.common.unittest as unittest
from pyomo.contrib.gdpopt.enumerate import GDP_Enumeration_Solver
from pyomo.environ import (
from pyomo.gdp import Disjunction
import pyomo.gdp.tests.models as models
def test_stop_at_iteration_limit(self):
    m = models.makeLogicalConstraintsOnDisjuncts()
    results = SolverFactory('gdpopt.enumerate').solve(m, iterlim=4, force_subproblem_nlp=True)
    self.assertEqual(results.solver.iterations, 4)
    self.assertEqual(results.solver.termination_condition, TerminationCondition.maxIterations)