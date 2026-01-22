import pyomo.common.unittest as unittest
import pyomo.environ as pyo
from pyomo.opt.base.solvers import UnknownSolver
from pyomo.core.plugins.transform.scaling import ScaleModel
def test_get_float_scaling_factor_local_level(self):
    m = pyo.ConcreteModel()
    m.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b1 = pyo.Block()
    m.b1.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.b1.b2 = pyo.Block()
    m.b1.b2.scaling_factor = pyo.Suffix(direction=pyo.Suffix.EXPORT)
    m.v1 = pyo.Var(initialize=10)
    m.b1.v2 = pyo.Var(initialize=20)
    m.b1.b2.v3 = pyo.Var(initialize=30)
    m.scaling_factor[m.v1] = 0.1
    m.b1.scaling_factor[m.b1.v2] = 0.2
    m.b1.b2.scaling_factor[m.b1.b2.v3] = 0.3
    m.b1.scaling_factor[m.b1.b2.v3] = 0.4
    sf = ScaleModel()._get_float_scaling_factor(m.v1)
    assert sf == float(0.1)
    sf = ScaleModel()._get_float_scaling_factor(m.b1.v2)
    assert sf == float(0.2)
    sf = ScaleModel()._get_float_scaling_factor(m.b1.b2.v3)
    assert sf == float(0.4)