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
def test_create_from_snapshot(self):
    stack_name = 'test_volume_create_from_snapshot_stack'
    fv = vt_base.FakeVolume('restoring-backup')
    fvbr = vt_base.FakeBackupRestore('vol-123')
    cinder.CinderClientPlugin._create.return_value = self.cinder_fc
    self.m_restore.return_value = fvbr
    fv2 = vt_base.FakeVolume('available')
    self.cinder_fc.volumes.get.side_effect = [fv, fv2]
    vol_name = utils.PhysName(stack_name, 'DataVolume')
    self.t['Resources']['DataVolume']['Properties']['SnapshotId'] = 'backup-123'
    stack = utils.parse_stack(self.t, stack_name=stack_name)
    self.create_volume(self.t, stack, 'DataVolume', no_create=True)
    cinder.CinderClientPlugin._create.assert_called_once_with()
    self.m_restore.assert_called_once_with('backup-123')
    self.cinder_fc.volumes.get.assert_called_with('vol-123')
    self.cinder_fc.volumes.update.assert_called_once_with('vol-123', description=vol_name, name=vol_name)