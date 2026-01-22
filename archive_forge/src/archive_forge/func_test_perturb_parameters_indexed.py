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
def test_perturb_parameters_indexed(self):
    delta = 1.0
    model = make_indexed_model()
    param_list = [model.eta]
    model.perturbed_eta = Param([1, 2], mutable=True, initialize={i: p.value + delta for i, p in model.eta.items()})
    ptb_list = [model.perturbed_eta]
    sens = SensitivityInterface(model, clone_model=False)
    sens.setup_sensitivity(param_list)
    sens.perturb_parameters(ptb_list)
    instance = sens.model_instance
    block = sens.block
    param_var_map = ComponentMap(((param, var) for var, param, _, _ in sens.block._sens_data_list))
    param_con_map = ComponentMap(((param, block.paramConst[i + 1]) for i, (_, param, _, _) in enumerate(sens.block._sens_data_list)))
    for param, ptb in zip(param_list, ptb_list):
        for idx in param:
            obj = param[idx]
            ptb_data = ptb[idx]
            var = param_var_map[obj]
            con = param_con_map[obj]
            self.assertEqual(instance.sens_state_value_1[var], ptb_data.value)
            self.assertEqual(instance.DeltaP[con], -delta)