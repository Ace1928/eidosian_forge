import copy
from unittest import mock
from cinderclient import exceptions as cinder_exp
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import cinder
from heat.engine.clients.os import nova
from heat.engine.resources.aws.ec2 import instance
from heat.engine.resources.aws.ec2 import volume as aws_vol
from heat.engine import rsrc_defn
from heat.engine import scheduler
from heat.tests.openstack.cinder import test_volume_utils as vt_base
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_volume_delete_error(self):
    fv = vt_base.FakeVolume('creating')
    stack_name = 'test_volume_deleting_stack'
    fv = self._mock_create_volume(vt_base.FakeVolume('creating'), stack_name)
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    rsrc = self.create_volume(self.t, stack, 'DataVolume')
    self.assertEqual(2, self.cinder_fc.volumes.get.call_count)
    self.cinder_fc.volumes.get.side_effect = [fv, vt_base.FakeVolume('deleting'), vt_base.FakeVolume('error_deleting')]
    self.cinder_fc.volumes.delete.return_value = True
    deleter = scheduler.TaskRunner(rsrc.destroy)
    self.assertRaisesRegex(exception.ResourceFailure, '.*ResourceInError.*error_deleting.*delete', deleter)
    self.cinder_fc.volumes.delete.assert_called_once_with(fv.id)
    self.assertEqual(5, self.cinder_fc.volumes.get.call_count)