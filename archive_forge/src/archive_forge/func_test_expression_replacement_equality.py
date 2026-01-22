import pyomo.common.unittest as unittest
from io import StringIO
import logging
from pyomo.environ import (
from pyomo.core.base.component import ComponentData
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap, ComponentSet
from pyomo.core.expr.visitor import identify_variables, identify_mutable_parameters
from pyomo.contrib.sensitivity_toolbox.sens import (
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_example
from pyomo.opt import SolverFactory
from pyomo.common.dependencies import (
from pyomo.common.dependencies import scipy_available
def test_expression_replacement_equality(self):
    model = make_indexed_model()
    sens = SensitivityInterface(model, clone_model=False)
    sens._add_data_block()
    instance = sens.model_instance
    block = sens.block
    instance.x.fix()
    param_list = [instance.eta[1], instance.eta[2]]
    sens._add_sensitivity_data(param_list)
    orig_components = list(instance.component_data_objects(Constraint, active=True)) + list(instance.component_data_objects(Objective, active=True))
    orig_expr = [con.expr for con in orig_components]
    expected_variables = ComponentMap(((con, ComponentSet(identify_variables(con.expr))) for con in orig_components))
    expected_parameters = ComponentMap(((con, ComponentSet(identify_mutable_parameters(con.expr))) for con in orig_components))
    variable_sub_map = dict(((id(param), var) for var, param, list_idx, _ in block._sens_data_list if param_list[list_idx].ctype is Param))
    self.assertEqual(len(variable_sub_map), 2)
    param_var_map = ComponentMap(((param, var) for var, param, _, _ in block._sens_data_list))
    for con in orig_components:
        for param in param_var_map:
            if param in expected_parameters[con]:
                expected_variables[con].add(param_var_map[param])
                expected_parameters[con].remove(param)
    replaced = sens._replace_parameters_in_constraints(variable_sub_map)
    self.assertEqual(len(block.constList), 2)
    for con in block.constList.values():
        self.assertTrue(con.active)
        param_set = ComponentSet(identify_mutable_parameters(con.expr))
        var_set = ComponentSet(identify_variables(con.expr))
        orig_con = replaced[con]
        self.assertIsNot(orig_con, con)
        self.assertEqual(param_set, expected_parameters[orig_con])
        self.assertEqual(var_set, expected_variables[orig_con])
    self.assertIs(block.cost.ctype, Objective)
    obj = block.cost
    param_set = ComponentSet(identify_mutable_parameters(obj.expr))
    var_set = ComponentSet(identify_variables(obj.expr))
    orig_obj = replaced[obj]
    self.assertIsNot(orig_obj, obj)
    self.assertEqual(param_set, expected_parameters[orig_obj])
    self.assertEqual(var_set, expected_variables[orig_obj])
    for con, expr in zip(orig_components, orig_expr):
        self.assertFalse(con.active)
        self.assertEqual(con.expr.to_string(), expr.to_string())