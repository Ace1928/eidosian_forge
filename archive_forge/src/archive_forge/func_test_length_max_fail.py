from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_length_max_fail(self):
    cl = constraints.Length(max=5, description='a range')
    self.assertRaises(ValueError, cl.validate, 'abcdef')