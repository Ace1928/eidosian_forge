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
def test_expression_replacement_no_replacement(self):
    model = make_indexed_model()
    sens = SensitivityInterface(model, clone_model=False)
    sens._add_data_block()
    instance = sens.model_instance
    block = sens.block
    instance.x.fix()
    param_list = [instance.x[1], instance.x[2], instance.x[3]]
    sens._add_sensitivity_data(param_list)
    self.assertEqual(len(block.constList), 0)
    variable_sub_map = {}
    sens._replace_parameters_in_constraints(variable_sub_map)
    self.assertEqual(len(block.constList), 2)
    pred_const_list = [instance.const[1], instance.const[2]]
    for orig, replaced in zip(pred_const_list, block.constList.values()):
        self.assertEqual(orig.expr.to_string(), replaced.expr.to_string())
        self.assertFalse(orig.active)
        self.assertTrue(replaced.active)