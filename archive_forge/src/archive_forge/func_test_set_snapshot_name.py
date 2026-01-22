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
def test_set_snapshot_name(self):
    snapshot_name = 'snapshot-name-' + uuid.uuid4().hex
    arglist = [self.share_snapshot.id, '--name', snapshot_name]
    verifylist = [('snapshot', self.share_snapshot.id), ('name', snapshot_name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.snapshots_mock.update.assert_called_with(self.share_snapshot, display_name=parsed_args.name)
    self.assertIsNone(result)