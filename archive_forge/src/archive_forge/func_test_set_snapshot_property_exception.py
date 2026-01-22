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
def test_set_snapshot_property_exception(self):
    arglist = ['--property', 'key=', self.share_snapshot.id]
    verifylist = [('property', {'key': ''}), ('snapshot', self.share_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.share_snapshot.set_metadata.assert_called_with({'key': ''})
    self.share_snapshot.set_metadata.side_effect = exceptions.BadRequest
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)