from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_unset(self):
    arglist = ['--property', 'foo', self.snapshot.id]
    verifylist = [('property', ['foo']), ('snapshot', self.snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.snapshots_mock.delete_metadata.assert_called_with(self.snapshot.id, ['foo'])
    self.assertIsNone(result)