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
@tools.stack_context('service_mark_unhealthy_lockexc_no_converge_test_stk')
def test_mark_unhealthy_stack_lock_exc_no_convergence(self):
    self.patchobject(stack_lock.StackLock, 'acquire', return_value=None, side_effect=exception.ActionInProgress(stack_name=self.stack.name, action=self.stack.action))
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.resource_mark_unhealthy, self.ctx, self.stack.identifier(), 'WebServer', True, resource_status_reason='')
    self.assertEqual(exception.ActionInProgress, ex.exc_info[0])