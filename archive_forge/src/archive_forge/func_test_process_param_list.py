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
def test_process_param_list(self):
    model = make_indexed_model()
    sens = SensitivityInterface(model, clone_model=False)
    param_list = [model.x[1], model.eta]
    new_param_list = sens._process_param_list(param_list)
    self.assertIs(param_list, new_param_list)
    sens = SensitivityInterface(model, clone_model=True)
    new_param_list = sens._process_param_list(param_list)
    self.assertIs(new_param_list[0], sens.model_instance.x[1])
    self.assertIs(new_param_list[1], sens.model_instance.eta)