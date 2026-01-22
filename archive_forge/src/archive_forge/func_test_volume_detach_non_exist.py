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
def test_volume_detach_non_exist(self):
    fv = vt_base.FakeVolume('creating')
    fva = vt_base.FakeVolume('in-use')
    stack_name = 'test_volume_detach_nonexist_stack'
    mock_attachment = self._mock_create_server_volume_script(fva)
    self._mock_create_volume(fv, stack_name, mock_attachment=mock_attachment)
    self.stub_VolumeConstraint_validate()
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    self.create_volume(self.t, stack, 'DataVolume')
    rsrc = self.create_attachment(self.t, stack, 'MountPoint')
    self.fc.volumes.delete_server_volume.return_value = None
    self.cinder_fc.volumes.get.side_effect = cinder_exp.NotFound('Not found')
    exc = fakes_nova.fake_exception()
    self.fc.volumes.get_server_volume.side_effect = exc
    scheduler.TaskRunner(rsrc.delete)()
    self.fc.volumes.delete_server_volume.assert_called_once_with(u'WikiDatabase', 'vol-123')
    self.fc.volumes.get_server_volume.assert_called_with(u'WikiDatabase', 'vol-123')
    self.validate_mock_create_server_volume_script()