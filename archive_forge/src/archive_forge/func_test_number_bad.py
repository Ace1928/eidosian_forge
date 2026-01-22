from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_number_bad(self):
    schema = {'Type': 'Number'}
    err = self.assertRaises(exception.StackValidationFailed, new_parameter, 'p', schema, 'str')
    self.assertIn('float', str(err))