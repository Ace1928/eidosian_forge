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
def test_param_const_indexed(self):
    model = make_indexed_model()
    param_list = [model.eta]
    sens = SensitivityInterface(model, clone_model=False)
    sens.setup_sensitivity(param_list)
    block = sens.block
    param_const = block.paramConst
    param_var_map = ComponentMap(((param, var) for var, param, _, _ in block._sens_data_list))
    for con in param_const.values():
        var_list = list(identify_variables(con.expr))
        mut_param_list = list(identify_mutable_parameters(con.expr))
        self.assertEqual(len(var_list), 1)
        self.assertEqual(len(mut_param_list), 1)
        self.assertIs(var_list[0], param_var_map[mut_param_list[0]])
        self.assertEqual(con.body.to_string(), (var_list[0] - mut_param_list[0]).to_string())