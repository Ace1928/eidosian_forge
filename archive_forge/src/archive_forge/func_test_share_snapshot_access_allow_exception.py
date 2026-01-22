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
def test_share_snapshot_access_allow_exception(self):
    arglist = [self.share_snapshot.id, 'user', 'demo']
    verifylist = [('snapshot', self.share_snapshot.id), ('access_type', 'user'), ('access_to', 'demo')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.snapshots_mock.allow.side_effect = exceptions.CommandError()
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)