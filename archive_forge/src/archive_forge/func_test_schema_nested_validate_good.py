from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_schema_nested_validate_good(self):
    nested = constraints.Schema(constraints.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Length(4, 8)])
    s = constraints.Schema(constraints.Schema.MAP, 'A map', schema={'Foo': nested})
    self.assertIsNone(s.validate())