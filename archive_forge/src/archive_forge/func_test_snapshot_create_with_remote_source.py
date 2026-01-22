from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_create_with_remote_source(self):
    arglist = ['--remote-source', 'source-name=test_source_name', '--remote-source', 'source-id=test_source_id', '--volume', self.new_snapshot.volume_id, self.new_snapshot.name]
    ref_dict = {'source-name': 'test_source_name', 'source-id': 'test_source_id'}
    verifylist = [('remote_source', ref_dict), ('volume', self.new_snapshot.volume_id), ('snapshot_name', self.new_snapshot.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.snapshots_mock.manage.assert_called_with(volume_id=self.new_snapshot.volume_id, ref=ref_dict, name=self.new_snapshot.name, description=None, metadata=None)
    self.snapshots_mock.create.assert_not_called()
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, data)