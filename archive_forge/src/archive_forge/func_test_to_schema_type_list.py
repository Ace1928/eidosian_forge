from heat.common import exception
from heat.engine import constraints
from heat.engine import environment
from heat.tests import common
def test_to_schema_type_list(self):
    """Test Schema.to_schema_type method for type List."""
    schema = constraints.Schema('List')
    res = schema.to_schema_type(['a', 'b'])
    self.assertIsInstance(res, list)
    self.assertEqual(['a', 'b'], res)