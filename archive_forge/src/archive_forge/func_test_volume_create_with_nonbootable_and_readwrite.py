from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume
@mock.patch.object(utils, 'wait_for_status', return_value=True)
def test_volume_create_with_nonbootable_and_readwrite(self, mock_wait):
    arglist = ['--non-bootable', '--read-write', '--size', str(self.new_volume.size), self.new_volume.name]
    verifylist = [('bootable', False), ('non_bootable', True), ('read_only', False), ('read_write', True), ('size', self.new_volume.size), ('name', self.new_volume.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_with(size=self.new_volume.size, snapshot_id=None, name=self.new_volume.name, description=None, volume_type=None, availability_zone=None, metadata=None, imageRef=None, source_volid=None, consistencygroup_id=None, scheduler_hints=None, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)
    self.volumes_mock.set_bootable.assert_called_with(self.new_volume.id, False)
    self.volumes_mock.update_readonly_flag.assert_called_with(self.new_volume.id, False)