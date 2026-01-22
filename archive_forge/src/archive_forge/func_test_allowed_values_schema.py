from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_allowed_values_schema(self):
    d = {'allowed_values': ['foo', 'bar'], 'description': 'allowed values'}
    r = constraints.AllowedValues(['foo', 'bar'], description='allowed values')
    self.assertEqual(d, dict(r))