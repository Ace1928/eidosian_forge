from unittest import mock
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as project_fakes
from openstackclient.tests.unit import utils as test_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_snapshot
def test_snapshot_set_no_option(self):
    arglist = [self.snapshot.id]
    verifylist = [('snapshot', self.snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.snapshots_mock.get.assert_called_once_with(parsed_args.snapshot)
    self.assertNotCalled(self.snapshots_mock.reset_state)
    self.assertNotCalled(self.snapshots_mock.update)
    self.assertNotCalled(self.snapshots_mock.set_metadata)
    self.assertIsNone(result)