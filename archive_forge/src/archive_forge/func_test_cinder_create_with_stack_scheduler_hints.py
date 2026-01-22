import collections
import copy
import json
from unittest import mock
from cinderclient import exceptions as cinder_exp
from novaclient import exceptions as nova_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import glance
from heat.engine.clients.os import nova
from heat.engine.resources.openstack.cinder import volume as c_vol
from heat.engine.resources import scheduler_hints as sh
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.objects import resource_data as resource_data_object
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_cinder_create_with_stack_scheduler_hints(self):
    fv = vt_base.FakeVolume('creating')
    sh.cfg.CONF.set_override('stack_scheduler_hints', True)
    self.stack_name = 'test_cvolume_stack_scheduler_hints_stack'
    t = template_format.parse(single_cinder_volume_template)
    stack = utils.parse_stack(t, stack_name=self.stack_name)
    rsrc = stack['volume']
    stack.add_resource(rsrc)
    self.assertIsNotNone(rsrc.uuid)
    shm = sh.SchedulerHintsMixin
    self.cinder_fc.volumes.create.return_value = fv
    fv_ready = vt_base.FakeVolume('available', id=fv.id)
    self.cinder_fc.volumes.get.side_effect = [fv, fv_ready]
    self.patchobject(rsrc, '_store_config_default_properties')
    scheduler.TaskRunner(rsrc.create)()
    self.assertGreater(rsrc.id, 0)
    self.cinder_fc.volumes.create.assert_called_once_with(size=1, name='test_name', description='test_description', availability_zone=None, metadata={}, scheduler_hints={shm.HEAT_ROOT_STACK_ID: stack.root_stack_id(), shm.HEAT_STACK_ID: stack.id, shm.HEAT_STACK_NAME: stack.name, shm.HEAT_PATH_IN_STACK: [stack.name], shm.HEAT_RESOURCE_NAME: rsrc.name, shm.HEAT_RESOURCE_UUID: rsrc.uuid})
    self.assertEqual(2, self.cinder_fc.volumes.get.call_count)