from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_length_invalid_type(self):
    schema = constraints.Schema('Integer', constraints=[constraints.Length(1, 10)])
    err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
    self.assertIn('Length constraint invalid for Integer', str(err))