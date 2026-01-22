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
@unittest.skipIf(not scipy_available, 'scipy is required for this test')
@unittest.skipIf(not opt.available(False), 'ipopt_sens is not available')
def test_indexedParamsMapping(self):
    m = hiv.create_model()
    hiv.initialize_model(m, 10, 5, 1)
    m.epsDelta = Param(initialize=0.75001)
    q_del = {}
    q_del[0, 0] = 1.001
    q_del[0, 1] = 1.002
    q_del[1, 0] = 1.003
    q_del[1, 1] = 1.004
    q_del[2, 0] = 0.83001
    q_del[2, 1] = 0.83002
    q_del[3, 0] = 0.42001
    q_del[4, 0] = 0.17001
    m.qqDelta = Param(m.ij, initialize=q_del)
    m.aaDelta = Param(initialize=0.0001001)
    m_sipopt = sensitivity_calculation('sipopt', m, [m.eps, m.qq, m.aa], [m.epsDelta, m.qqDelta, m.aaDelta])
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].lower, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].upper, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[1].body.to_string(), '_SENSITIVITY_TOOLBOX_DATA.eps - eps')
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].lower, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].upper, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[6].body.to_string(), '_SENSITIVITY_TOOLBOX_DATA.qq[2,0] - qq[2,0]')
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].lower, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].upper, 0.0)
    self.assertEqual(m_sipopt._SENSITIVITY_TOOLBOX_DATA.paramConst[10].body.to_string(), '_SENSITIVITY_TOOLBOX_DATA.aa - aa')