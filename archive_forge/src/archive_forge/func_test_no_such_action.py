import os.path
import ddt
from oslo_config import fixture as config_fixture
from oslo_policy import policy as base_policy
from heat.common import exception
from heat.common import policy
from heat.tests import common
from heat.tests import utils
def test_no_such_action(self):
    ctx = utils.dummy_context(roles=['not_a_stack_user'])
    enforcer = policy.Enforcer(scope='cloudformation')
    action = 'no_such_action'
    msg = 'cloudformation:no_such_action has not been registered'
    self.assertRaisesRegex(base_policy.PolicyNotRegistered, msg, enforcer.enforce, ctx, action, None, None, True)