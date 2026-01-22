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
def test_volume_create_with_snapshot(self):
    snapshot = volume_fakes.create_one_snapshot()
    self.new_volume.snapshot_id = snapshot.id
    arglist = ['--snapshot', self.new_volume.snapshot_id, self.new_volume.name]
    verifylist = [('snapshot', self.new_volume.snapshot_id), ('name', self.new_volume.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.snapshots_mock.get.return_value = snapshot
    columns, data = self.cmd.take_action(parsed_args)
    self.volumes_mock.create.assert_called_once_with(size=snapshot.size, snapshot_id=snapshot.id, name=self.new_volume.name, description=None, volume_type=None, availability_zone=None, metadata=None, imageRef=None, source_volid=None, consistencygroup_id=None, scheduler_hints=None, backup_id=None)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)