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
def test_vaildate_deletion_policy(self):
    cfg.CONF.set_override('backups_enabled', False, group='volumes')
    self.stack_name = 'test_volume_validate_deletion_policy'
    self.t['resources']['volume']['deletion_policy'] = 'Snapshot'
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    rsrc = self.get_volume(self.t, stack, 'volume')
    self.assertRaisesRegex(exception.StackValidationFailed, 'volume backup service is not enabled', rsrc.validate)