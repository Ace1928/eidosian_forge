from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_modulo_schema(self):
    d = {'modulo': {'step': 2, 'offset': 1}, 'description': 'a modulo'}
    r = constraints.Modulo(2, 1, description='a modulo')
    self.assertEqual(d, dict(r))