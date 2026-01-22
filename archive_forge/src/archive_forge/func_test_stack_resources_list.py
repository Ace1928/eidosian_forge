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
@mock.patch.object(stack.Stack, 'load')
@tools.stack_context('service_resources_list_test_stack')
def test_stack_resources_list(self, mock_load):
    mock_load.return_value = self.stack
    resources = self.eng.list_stack_resources(self.ctx, self.stack.identifier())
    self.assertEqual(1, len(resources))
    r = resources[0]
    self.assertIn('resource_identity', r)
    self.assertIn('updated_time', r)
    self.assertIn('physical_resource_id', r)
    self.assertIn('resource_name', r)
    self.assertEqual('WebServer', r['resource_name'])
    self.assertIn('resource_status', r)
    self.assertIn('resource_status_reason', r)
    self.assertIn('resource_type', r)
    mock_load.assert_called_once_with(self.ctx, stack=mock.ANY)