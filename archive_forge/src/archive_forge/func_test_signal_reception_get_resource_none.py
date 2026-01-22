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
def test_signal_reception_get_resource_none(self):
    stack_name = 'signal_reception_no_resource'
    self.stack = self._stack_create(stack_name)
    test_data = {'food': 'yum'}
    self.patchobject(stack.Stack, 'resource_get', return_value=None)
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.resource_signal, self.ctx, dict(self.stack.identifier()), 'WebServerScaleDownPolicy', test_data)
    self.assertEqual(exception.ResourceNotFound, ex.exc_info[0])