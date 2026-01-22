from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_schema_validate_fail(self):
    s = constraints.Schema(constraints.Schema.STRING, 'A string', default='wibble', constraints=[constraints.Range(max=4)])
    err = self.assertRaises(exception.InvalidSchemaError, s.validate)
    self.assertIn('Range constraint invalid for String', str(err))