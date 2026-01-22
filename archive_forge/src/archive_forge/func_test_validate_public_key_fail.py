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
def test_validate_public_key_fail(self):
    self.patchobject(nova.NovaClientPlugin, 'get_max_microversion', return_value='2.92')
    template = copy.deepcopy(self.kp_template)
    stack = utils.parse_stack(template)
    definition = stack.t.resource_definitions(stack)['kp']
    kp_res = keypair.KeyPair('kp', definition, stack)
    error = self.assertRaises(exception.StackValidationFailed, kp_res.validate)
    msg = 'The public_key property is required by the nova API version currently used.'
    self.assertIn(msg, str(error))