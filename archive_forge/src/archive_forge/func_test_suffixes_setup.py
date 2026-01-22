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
def test_suffixes_setup(self):
    model = make_indexed_model()
    param_list = [model.eta[2], model.eta[1]]
    sens = SensitivityInterface(model, clone_model=False)
    sens.setup_sensitivity(param_list)
    for i, (var, _, _, _) in enumerate(sens.block._sens_data_list):
        con = sens.block.paramConst[i + 1]
        self.assertEqual(model.sens_state_0[var], i + 1)
        self.assertEqual(model.sens_state_1[var], i + 1)
        self.assertEqual(model.sens_init_constr[con], i + 1)
        self.assertEqual(model.dcdp[con], i + 1)
    self.assertIs(type(model.sens_sol_state_1_z_L), Suffix)
    self.assertIs(type(model.sens_sol_state_1_z_U), Suffix)
    self.assertIs(type(model.ipopt_zL_out), Suffix)
    self.assertIs(type(model.ipopt_zU_out), Suffix)
    self.assertIs(type(model.ipopt_zL_in), Suffix)
    self.assertIs(type(model.ipopt_zU_in), Suffix)
    self.assertIs(type(model.dual), Suffix)
    self.assertIs(type(model.DeltaP), Suffix)