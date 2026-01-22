from unittest import mock
from keystoneauth1 import discover
from openstack.block_storage.v3 import group as _group
from openstack.block_storage.v3 import group_snapshot as _group_snapshot
from openstack.test import fakes as sdk_fakes
from openstack import utils as sdk_utils
from osc_lib import exceptions
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
from openstackclient.volume.v3 import volume_group_snapshot
@mock.patch.object(sdk_utils, 'supports_microversion')
def test_volume_group_snapshot_create(self, mock_mv):
    mock_mv.side_effect = fake_supports_microversion('3.14')
    arglist = [self.fake_volume_group.id]
    verifylist = [('volume_group', self.fake_volume_group.id), ('name', None), ('description', None)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volume_sdk_client.find_group.assert_called_once_with(self.fake_volume_group.id, ignore_missing=False, details=False)
    self.volume_sdk_client.create_group_snapshot.assert_called_once_with(group_id=self.fake_volume_group.id, name=None, description=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)