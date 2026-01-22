from unittest import mock
from heat.common import exception
from heat.engine.cfn import functions as cfn_funcs
from heat.engine.clients.os import monasca as client_plugin
from heat.engine.resources.openstack.monasca import notification
from heat.engine import stack
from heat.engine import template
from heat.tests import common
from heat.tests import utils
def test_validate_invalid_address_parsing(self):
    self.test_resource.properties.data['type'] = self.test_resource.WEBHOOK
    self.test_resource.properties.data['address'] = 'https://example.com]'
    ex = self.assertRaises(exception.StackValidationFailed, self.test_resource.validate)
    self.assertEqual('Address "https://example.com]" should have correct format required by "webhook" type of "type" property', str(ex))