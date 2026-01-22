from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_string(self):
    """Test Schema.to_schema_type method for type String."""
    schema = constraints.Schema('String')
    res = schema.to_schema_type('one')
    self.assertIsInstance(res, str)
    res = schema.to_schema_type('1')
    self.assertIsInstance(res, str)
    res = schema.to_schema_type(1)
    self.assertIsInstance(res, str)
    res = schema.to_schema_type(True)
    self.assertIsInstance(res, str)
    res = schema.to_schema_type(None)
    self.assertIsInstance(res, str)