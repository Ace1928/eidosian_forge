from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_length_min_schema(self):
    d = {'length': {'min': 5}, 'description': 'a length range'}
    r = constraints.Length(min=5, description='a length range')
    self.assertEqual(d, dict(r))