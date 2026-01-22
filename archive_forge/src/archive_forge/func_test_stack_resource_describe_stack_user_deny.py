from unittest import mock
from oslo_config import cfg
from oslo_messaging.rpc import dispatcher
from heat.common import exception
from heat.common import identifier
from heat.engine.clients.os import heat_plugin
from heat.engine.clients.os import keystone
from heat.engine.clients.os.keystone import fake_keystoneclient as fake_ks
from heat.engine import dependencies
from heat.engine import resource as res
from heat.engine.resources.aws.ec2 import instance as ins
from heat.engine import service
from heat.engine import stack
from heat.engine import stack_lock
from heat.engine import template as templatem
from heat.objects import stack as stack_object
from heat.tests import common
from heat.tests.engine import tools
from heat.tests import generic_resource as generic_rsrc
from heat.tests import utils
@mock.patch.object(service.EngineService, '_authorize_stack_user')
@tools.stack_context('service_resource_describe_user_deny_test_stack')
def test_stack_resource_describe_stack_user_deny(self, mock_auth):
    self.ctx.roles = [cfg.CONF.heat_stack_user_role]
    mock_auth.return_value = False
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.describe_stack_resource, self.ctx, self.stack.identifier(), 'foo')
    self.assertEqual(exception.Forbidden, ex.exc_info[0])
    mock_auth.assert_called_once_with(self.ctx, mock.ANY, 'foo')