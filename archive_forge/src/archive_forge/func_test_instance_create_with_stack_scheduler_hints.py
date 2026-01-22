import copy
from unittest import mock
import uuid
from neutronclient.v2_0 import client as neutronclient
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients import progress
from heat.engine import environment
from heat.engine import resource
from heat.engine.resources.aws.ec2 import instance as instances
from heat.engine.resources import scheduler_hints as sh
from heat.engine import scheduler
from heat.engine import stack as parser
from heat.engine import template
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_instance_create_with_stack_scheduler_hints(self):
    return_server = self.fc.servers.list()[1]
    sh.cfg.CONF.set_override('stack_scheduler_hints', True)
    stack_name = 'test_instance_create_with_stack_scheduler_hints'
    t, stack = self._get_test_template(stack_name)
    resource_defns = t.resource_definitions(stack)
    instance = instances.Instance('in_create_with_sched_hints', resource_defns['WebServer'], stack)
    bdm = {'vdb': '9ef5496e-7426-446a-bbc8-01f84d9c9972:snap::True'}
    self._mock_get_image_id_success('CentOS 5.2', 1)
    stack.add_resource(instance)
    self.assertIsNotNone(instance.uuid)
    self.patchobject(nova.NovaClientPlugin, 'client', return_value=self.fc)
    self.stub_SnapshotConstraint_validate()
    self.fc.servers.create = mock.Mock(return_value=return_server)
    scheduler.TaskRunner(instance.create)()
    self.assertGreater(instance.id, 0)
    shm = sh.SchedulerHintsMixin
    self.fc.servers.create.assert_called_once_with(image=1, flavor=1, key_name='test', name=utils.PhysName(stack_name, instance.name, limit=instance.physical_resource_name_limit), security_groups=None, userdata=mock.ANY, scheduler_hints={shm.HEAT_ROOT_STACK_ID: stack.root_stack_id(), shm.HEAT_STACK_ID: stack.id, shm.HEAT_STACK_NAME: stack.name, shm.HEAT_PATH_IN_STACK: [stack.name], shm.HEAT_RESOURCE_NAME: instance.name, shm.HEAT_RESOURCE_UUID: instance.uuid, 'foo': ['spam', 'ham', 'baz'], 'bar': 'eggs'}, meta=None, nics=None, availability_zone=None, block_device_mapping=bdm)