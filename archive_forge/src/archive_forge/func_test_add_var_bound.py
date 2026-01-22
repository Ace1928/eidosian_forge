import pyomo.common.unittest as unittest
from pyomo.contrib.mindtpy.util import set_var_valid_value
from pyomo.environ import Var, Integers, ConcreteModel, Integers
from pyomo.contrib.mindtpy.algorithm_base_class import _MindtPyAlgorithm
from pyomo.contrib.mindtpy.config_options import _get_MindtPy_OA_config
from pyomo.contrib.mindtpy.tests.MINLP5_simple import SimpleMINLP5
from pyomo.contrib.mindtpy.util import add_var_bound
def test_add_var_bound(self):
    m = SimpleMINLP5().clone()
    m.x.lb = None
    m.x.ub = None
    m.y.lb = None
    m.y.ub = None
    solver_object = _MindtPyAlgorithm()
    solver_object.config = _get_MindtPy_OA_config()
    solver_object.set_up_solve_data(m)
    solver_object.create_utility_block(solver_object.working_model, 'MindtPy_utils')
    add_var_bound(solver_object.working_model, solver_object.config)
    self.assertEqual(solver_object.working_model.x.lower, -solver_object.config.continuous_var_bound - 1)
    self.assertEqual(solver_object.working_model.x.upper, solver_object.config.continuous_var_bound)
    self.assertEqual(solver_object.working_model.y.lower, -solver_object.config.integer_var_bound - 1)
    self.assertEqual(solver_object.working_model.y.upper, solver_object.config.integer_var_bound)