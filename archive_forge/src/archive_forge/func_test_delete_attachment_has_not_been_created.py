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
def test_delete_attachment_has_not_been_created(self):
    self.stack_name = 'test_delete_attachment_has_not_been_created'
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    resource_defn = stack.t.resource_definitions(stack)
    att_rsrc = c_vol.CinderVolumeAttachment('test_attachment', resource_defn['attachment'], stack)
    att_rsrc.state_set(att_rsrc.UPDATE, att_rsrc.COMPLETE)
    self.assertIsNone(att_rsrc.resource_id)
    nc = self.patchobject(nova.NovaClientPlugin, '_create')
    scheduler.TaskRunner(att_rsrc.delete)()
    self.assertEqual(0, nc.call_count)
    self.assertEqual((att_rsrc.DELETE, att_rsrc.COMPLETE), att_rsrc.state)