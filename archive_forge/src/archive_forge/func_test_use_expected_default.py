from oslo_serialization import jsonutils as json
from heat.common import exception
from heat.common import identifier
from heat.engine import parameters
from heat.engine import template
from heat.tests import common
def test_use_expected_default(self):
    template = {'Parameters': {'a': {'Type': self.p_type, 'Default': self.default}}}
    params = self.new_parameters('test_params', template)
    self.assertEqual(self.expected[0], params['a'])
    params = self.new_parameters('test_params', template, param_defaults={'a': self.param_default})
    self.assertEqual(self.expected[1], params['a'])
    params = self.new_parameters('test_params', template, {'a': self.value}, param_defaults={'a': self.param_default})
    self.assertEqual(self.expected[2], params['a'])