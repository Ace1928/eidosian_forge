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
@unittest.skipIf(not opt_kaug.available(False), 'k_aug is not available')
@unittest.skipIf(not opt_dotsens.available(False), 'dot_sens is not available')
@unittest.skipIf(not scipy_available, 'scipy is not available')
def test_get_dfds_dcds2(self):
    """
        It tests the function get_sensitivity with rooney & biegler's model.
        """
    variable_name = ['asymptote', 'rate_constant']
    theta = {'asymptote': 19.142575284617866, 'rate_constant': 0.53109137696521}
    cov = np.array([[6.30579403, -0.4395341], [-0.4395341, 0.04193591]])
    model_uncertain = ConcreteModel()
    model_uncertain.asymptote = Var(initialize=15)
    model_uncertain.rate_constant = Var(initialize=0.5)
    model_uncertain.obj = Objective(expr=model_uncertain.asymptote * (1 - exp(-model_uncertain.rate_constant * 10)), sense=minimize)
    theta = {'asymptote': 19.142575284617866, 'rate_constant': 0.53109137696521}
    for v in variable_name:
        getattr(model_uncertain, v).setlb(theta[v])
        getattr(model_uncertain, v).setub(theta[v])
    gradient_f, gradient_c, col, row, line_dic = get_dfds_dcds(model_uncertain, variable_name)
    np.testing.assert_almost_equal(gradient_f, [0.99506259, 0.945148])
    np.testing.assert_almost_equal(gradient_c, np.array([]))
    assert col == ['asymptote', 'rate_constant']
    assert row == ['obj']