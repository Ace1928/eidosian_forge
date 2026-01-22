from io import StringIO
import logging
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Param, Var, Block, Suffix, value
from pyomo.opt import SolverFactory
from pyomo.dae import ContinuousSet
from pyomo.common.dependencies import scipy_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.collections import ComponentMap
from pyomo.core.expr import identify_variables
from pyomo.contrib.sensitivity_toolbox.sens import sipopt, kaug, sensitivity_calculation
import pyomo.contrib.sensitivity_toolbox.examples.parameter as param_ex
import pyomo.contrib.sensitivity_toolbox.examples.parameter_kaug as param_kaug_ex
import pyomo.contrib.sensitivity_toolbox.examples.feedbackController as fc
import pyomo.contrib.sensitivity_toolbox.examples.rangeInequality as ri
import pyomo.contrib.sensitivity_toolbox.examples.HIV_Transmission as hiv
@unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
def test_sipopt_deprecated(self):
    m = param_ex.create_model()
    m.perturbed_eta1 = Param(initialize=4.0)
    m.perturbed_eta2 = Param(initialize=1.0)
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.sensitivity_toolbox', logging.WARNING):
        sipopt(m, [m.eta1, m.eta1], [m.perturbed_eta1, m.perturbed_eta2], cloneModel=False)
    self.assertIn("DEPRECATED: The sipopt function has been deprecated. Use the sensitivity_calculation() function with method='sipopt' to access", output.getvalue().replace('\n', ' '))