import pyomo.common.unittest as unittest
from pyomo.environ import ConcreteModel, Var, Integers, value
from pyomo.environ import TransformationFactory as xfrm
from pyomo.common.log import LoggingIntercept
import logging
from io import StringIO
def test_no_integer(self):
    m = ConcreteModel()
    m.x = Var()
    output = StringIO()
    with LoggingIntercept(output, 'pyomo.contrib.preprocessing', logging.INFO):
        xfrm('contrib.integer_to_binary').apply_to(m)
    expected_message = 'Model has no free integer variables.'
    self.assertIn(expected_message, output.getvalue())