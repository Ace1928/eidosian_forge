from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_unknown_params(self):
    user_params = {'Foo': 'wibble'}
    self.assertRaises(exception.UnknownUserParameter, self.new_parameters, 'test', params_schema, user_params)