import logging
from math import pi
from io import StringIO
import pyomo.common.unittest as unittest
from pyomo.common.log import LoggingIntercept
from pyomo.common.dependencies.matplotlib import pyplot as plt
from pyomo.contrib.mcpp.pyomo_mcpp import McCormick as mc, mcpp_available, MCPP_Error
from pyomo.core import (
from pyomo.core.expr import identify_variables
def test_lmtd(self):
    m = ConcreteModel()
    m.x = Var(bounds=(0.1, 500), initialize=33.327)
    m.y = Var(bounds=(0.1, 500), initialize=14.436)
    m.z = Var(bounds=(0, 90), initialize=22.5653)
    e = m.z - (m.x * m.y * (m.x + m.y) / 2) ** (1 / 3)
    mc_expr = mc(e)
    for _x in [m.x.lb, m.x.ub]:
        m.x.value = _x
        mc_expr.changePoint(m.x, _x)
        for _y in [m.y.lb, m.y.ub]:
            m.y.value = _y
            mc_expr.changePoint(m.y, _y)
            for _z in [m.z.lb, m.z.ub]:
                m.z.value = _z
                mc_expr.changePoint(m.z, _z)
                self.assertGreaterEqual(mc_expr.concave() + 1e-08, value(e))
                self.assertLessEqual(mc_expr.convex() - 1e-06, value(e))
    m.x.value = m.x.lb
    m.y.value = m.y.lb
    m.z.value = m.z.lb
    mc_expr.changePoint(m.x, m.x.value)
    mc_expr.changePoint(m.y, m.y.value)
    mc_expr.changePoint(m.z, m.z.value)
    self.assertAlmostEqual(mc_expr.convex(), value(e))
    self.assertAlmostEqual(mc_expr.concave(), value(e))
    m.x.value = m.x.ub
    m.y.value = m.y.ub
    m.z.value = m.z.ub
    mc_expr.changePoint(m.x, m.x.value)
    mc_expr.changePoint(m.y, m.y.value)
    mc_expr.changePoint(m.z, m.z.value)
    self.assertAlmostEqual(mc_expr.convex(), value(e))
    self.assertAlmostEqual(mc_expr.concave(), value(e))
    self.assertAlmostEqual(mc_expr.lower(), -500)
    self.assertAlmostEqual(mc_expr.upper(), 89.9)