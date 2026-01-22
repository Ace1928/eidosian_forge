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
def test_cinder_create_with_image_and_imageRef(self):
    self.stack_name = 'test_create_with_image_and_imageRef'
    combinations = {'imageRef': 'image-456', 'image': 'image-123'}
    err_msg = 'Cannot define the following properties at the same time: image, imageRef'
    self.stub_ImageConstraint_validate()
    stack = utils.parse_stack(self.t, stack_name=self.stack_name)
    vp = stack.t['Resources']['volume2']['Properties']
    vp.pop('size')
    vp.update(combinations)
    rsrc = stack.get('volume2')
    ex = self.assertRaises(exception.StackValidationFailed, rsrc.validate)
    self.assertIn(err_msg, str(ex))