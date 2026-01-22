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
def test_cinder_create_with_image_and_size(self):
    self.stack_name = 'test_create_with_image_and_size'
    combinations = {'image': 'image-123'}
    err_msg = 'If neither "backup_id" nor "size" is provided, one and only one of "source_volid", "snapshot_id" must be specified, but currently specified options: [\'image\'].'
    self.stub_ImageConstraint_validate()
    self._test_cinder_create_invalid_property_combinations(self.stack_name, combinations, err_msg, exception.StackValidationFailed)