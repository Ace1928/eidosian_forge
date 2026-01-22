import argparse
import ddt
from unittest import mock
import uuid
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.api_versions import MAX_VERSION
from manilaclient.common.apiclient import exceptions
from manilaclient.common import cliutils
from manilaclient.osc.v2 import share as osc_shares
from manilaclient.tests.unit.osc import osc_fakes
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_create_with_snapshot(self):
    """Verifies create share from snapshot."""
    arglist = [self.new_share.share_proto, str(self.new_share.size), '--snapshot-id', self.share_snapshot.id]
    verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('snapshot_id', self.share_snapshot.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('manilaclient.common.apiclient.utils.find_resource', mock.Mock(return_value=self.share_snapshot)):
        columns, data = self.cmd.take_action(parsed_args)
        osc_shares.apiutils.find_resource.assert_called_once_with(mock.ANY, self.share_snapshot.id)
    self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=None, size=self.new_share.size, snapshot_id=self.share_snapshot.id, scheduler_hints={})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, data)