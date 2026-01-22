import copy
from unittest import mock
from heat.common import exception
from heat.engine.clients.os import keystone
from heat.engine.clients.os import nova
from heat.engine import resource
from heat.engine.resources.openstack.nova import keypair
from heat.engine import scheduler
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_create_key_excess_name_length(self):
    """Test creation of a keypair whose name is of excess length."""
    key_name = 'k' * 256
    template = copy.deepcopy(self.kp_template)
    template['resources']['kp']['properties']['name'] = key_name
    stack = utils.parse_stack(template)
    definition = stack.t.resource_definitions(stack)['kp']
    kp_res = keypair.KeyPair('kp', definition, stack)
    error = self.assertRaises(exception.StackValidationFailed, kp_res.validate)
    self.assertIn('Property error', str(error))
    self.assertIn('kp.properties.name: length (256) is out of range (min: 1, max: 255)', str(error))