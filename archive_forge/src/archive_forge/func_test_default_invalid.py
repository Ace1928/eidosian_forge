from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_default_invalid(self):
    schema = {'Type': self.p_type, 'AllowedValues': self.allowed_value, 'ConstraintDescription': 'wibble', 'Default': self.default}
    if self.p_type == 'Json':
        err = self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', schema)
        self.assertIn('AllowedValues constraint invalid for Json', str(err))
    else:
        err = self.assertRaises(exception.InvalidSchemaError, new_parameter, 'p', schema)
        self.assertIn('wibble', str(err))