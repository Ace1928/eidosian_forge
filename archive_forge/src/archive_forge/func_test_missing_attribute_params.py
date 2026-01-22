from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_missing_attribute_params(self):
    params = {'Parameters': {'Foo': {'Type': 'String'}, 'NoAttr': 'No attribute.', 'Bar': {'Type': 'Number', 'Default': '1'}}}
    self.assertRaises(exception.InvalidSchemaError, self.new_parameters, 'test', params)