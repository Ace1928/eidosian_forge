import sys
import pyomo.common.unittest as unittest
from pyomo.environ import (
from pyomo.opt import SolverFactory, TerminationCondition, SolutionStatus
from pyomo.solvers.plugins.solvers.cplex_direct import (
def test_add_block_containing_multiple_variables(self):
    """Test that:
        - The variable is added correctly to `solver_model`
        - The CPLEX `variables` interface is called only once
        - Fixed variable bounds are set correctly
        """
    model = ConcreteModel()
    opt = SolverFactory('cplex', solver_io='python')
    opt._set_instance(model)
    self.assertEqual(opt._solver_model.variables.get_num(), 0)
    model.X1 = Var(within=Binary)
    model.X2 = Var(within=NonNegativeReals)
    model.X3 = Var(within=NonNegativeIntegers)
    model.X3.fix(5)
    with unittest.mock.patch.object(opt._solver_model.variables, 'add', wraps=opt._solver_model.variables.add) as wrapped_add_call:
        opt._add_block(model)
        self.assertEqual(wrapped_add_call.call_count, 1)
        self.assertEqual(wrapped_add_call.call_args, ({'lb': [0, 0, 5], 'names': ['x1', 'x2', 'x3'], 'types': ['B', 'C', 'I'], 'ub': [1, cplex.infinity, 5]},))
    self.assertEqual(opt._solver_model.variables.get_num(), 3)