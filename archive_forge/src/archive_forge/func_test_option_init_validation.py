import uuid
from keystone.common import resource_options
from keystone.tests import unit
def test_option_init_validation(self):
    self.assertRaises(TypeError, resource_options.ResourceOption, 'test', 1234)
    self.assertRaises(TypeError, resource_options.ResourceOption, 1234, 'testing')
    self.assertRaises(ValueError, resource_options.ResourceOption, 'testing', 'testing')
    resource_options.ResourceOption('test', 'testing')