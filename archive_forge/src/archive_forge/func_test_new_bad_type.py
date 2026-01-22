from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_new_bad_type(self):
    self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', {'Type': 'List'}, validate_value=False)