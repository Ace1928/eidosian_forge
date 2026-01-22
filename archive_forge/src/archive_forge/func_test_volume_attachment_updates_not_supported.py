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
def test_volume_attachment_updates_not_supported(self):
    self.patchobject(nova.NovaClientPlugin, 'get_server')
    fv = vt_base.FakeVolume('creating')
    fva = vt_base.FakeVolume('attaching')
    stack_name = 'test_volume_attach_updnotsup_stack'
    mock_create_server_volume = self._mock_create_server_volume_script(fva)
    self._mock_create_volume(fv, stack_name, mock_attachment=mock_create_server_volume)
    self.stub_VolumeConstraint_validate()
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    self.create_volume(self.t, stack, 'DataVolume')
    rsrc = self.create_attachment(self.t, stack, 'MountPoint')
    props = copy.deepcopy(rsrc.properties.data)
    props['InstanceId'] = 'some_other_instance_id'
    props['VolumeId'] = 'some_other_volume_id'
    props['Device'] = '/dev/vdz'
    after = rsrc_defn.ResourceDefinition(rsrc.name, rsrc.type(), props)
    update_task = scheduler.TaskRunner(rsrc.update, after)
    ex = self.assertRaises(exception.ResourceFailure, update_task)
    self.assertIn('NotSupported: resources.MountPoint: Update to properties Device, InstanceId, VolumeId of MountPoint (AWS::EC2::VolumeAttachment)', str(ex))
    self.assertEqual((rsrc.UPDATE, rsrc.FAILED), rsrc.state)
    self.validate_mock_create_server_volume_script()