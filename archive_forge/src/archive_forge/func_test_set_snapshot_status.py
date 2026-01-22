from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share_snapshots as osc_share_snapshots
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_snapshot_status(self):
    arglist = [self.share_snapshot.id, '--status', 'available']
    verifylist = [('snapshot', self.share_snapshot.id), ('status', 'available')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.snapshots_mock.reset_state.assert_called_with(self.share_snapshot, parsed_args.status)
    self.assertIsNone(result)