import os
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, AbstractModel, Set
from pyomo.dae import ContinuousSet
from pyomo.common.log import LoggingIntercept
from io import StringIO
def test_get_lower_element_boundary(self):
    m = ConcreteModel()
    m.t = ContinuousSet(initialize=[1, 2, 3])
    self.assertEqual(m.t.get_lower_element_boundary(1.5), 1)
    self.assertEqual(m.t.get_lower_element_boundary(2.5), 2)
    self.assertEqual(m.t.get_lower_element_boundary(2), 2)
    log_out = StringIO()
    with LoggingIntercept(log_out, 'pyomo.dae'):
        temp = m.t.get_lower_element_boundary(0.5)
    self.assertIn('Returning the lower bound', log_out.getvalue())