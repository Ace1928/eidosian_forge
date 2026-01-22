from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_list_value_list_bad(self):
    schema = {'Type': 'CommaDelimitedList', 'ConstraintDescription': 'wibble', 'AllowedValues': ['foo', 'bar', 'baz']}
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'foo,baz,blarg')
    self.assertIn('wibble', str(err))