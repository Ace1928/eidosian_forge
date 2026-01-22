from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_modulo_invalid_type(self):
    schema = constraints.Schema('String', constraints=[constraints.Modulo(2, 1)])
    err = self.assertRaises(exception.InvalidSchemaError, schema.validate)
    self.assertIn('Modulo constraint invalid for String', str(err))