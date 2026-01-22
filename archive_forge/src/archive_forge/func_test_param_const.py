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
def test_param_const(self):
    model = make_indexed_model()
    param_list = [model.eta[1], model.eta[2]]
    sens = SensitivityInterface(model, clone_model=False)
    sens.setup_sensitivity(param_list)
    block = sens.block
    param_const = block.paramConst
    self.assertEqual(len(param_list), len(block.paramConst))
    param_var_map = ComponentMap(((param, var) for var, param, _, _ in block._sens_data_list))
    var_list = [param_var_map[param] for param in param_list]
    for param, var, con in zip(param_list, var_list, param_const.values()):
        self.assertEqual(con.body.to_string(), (var - param).to_string())