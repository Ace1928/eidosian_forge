from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_param_name_in_error_message(self):
    schema = {'Type': 'String', 'AllowedPattern': '[a-z]*'}
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'testparam', schema, '234')
    expected = 'Parameter \'testparam\' is invalid: "234" does not match pattern "[a-z]*"'
    self.assertEqual(expected, str(err))