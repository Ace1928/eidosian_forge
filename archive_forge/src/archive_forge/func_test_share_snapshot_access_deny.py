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
def test_share_snapshot_access_deny(self):
    arglist = [self.share_snapshot.id, self.access_rule.id]
    verifylist = [('snapshot', self.share_snapshot.id), ('id', [self.access_rule.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.snapshots_mock.deny.assert_called_with(snapshot=self.share_snapshot, id=self.access_rule.id)
    self.assertIsNone(result)