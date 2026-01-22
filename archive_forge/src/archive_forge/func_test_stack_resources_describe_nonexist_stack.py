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
def test_stack_resources_describe_nonexist_stack(self):
    non_exist_identifier = identifier.HeatIdentifier(self.ctx.tenant_id, 'wibble', '18d06e2e-44d3-4bef-9fbf-52480d604b02')
    ex = self.assertRaises(dispatcher.ExpectedException, self.eng.describe_stack_resources, self.ctx, non_exist_identifier, 'WebServer')
    self.assertEqual(exception.EntityNotFound, ex.exc_info[0])