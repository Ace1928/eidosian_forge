import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_resource_enforce_returns_false(self):
    context = utils.dummy_context(roles=['non-admin'])
    enforcer = policy.ResourceEnforcer(exc=None)
    res_type = 'OS::Keystone::User'
    self.assertFalse(enforcer.enforce(context, res_type, is_registered_policy=True))
    self.assertIsNotNone(enforcer.enforce(context, res_type, is_registered_policy=True))