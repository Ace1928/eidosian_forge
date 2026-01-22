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
def test_unset_snapshot_property(self):
    arglist = ['--property', 'Manila', self.share_snapshot.id]
    verifylist = [('property', ['Manila']), ('snapshot', self.share_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.share_snapshot.delete_metadata.assert_called_with(parsed_args.property)