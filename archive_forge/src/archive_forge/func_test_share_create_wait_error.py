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
@mock.patch('manilaclient.osc.v2.share.LOG')
def test_share_create_wait_error(self, mock_logger):
    arglist = [self.new_share.share_proto, str(self.new_share.size), '--share-type', self.share_type.id, '--wait']
    verifylist = [('share_proto', self.new_share.share_proto), ('size', self.new_share.size), ('share_type', self.share_type.id), ('wait', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.wait_for_status', return_value=False):
        columns, data = self.cmd.take_action(parsed_args)
        self.shares_mock.create.assert_called_with(availability_zone=None, description=None, is_public=False, metadata={}, name=None, share_group_id=None, share_network=None, share_proto=self.new_share.share_proto, share_type=self.share_type.id, size=self.new_share.size, snapshot_id=None, scheduler_hints={})
        mock_logger.error.assert_called_with('ERROR: Share is in error state.')
        self.shares_mock.get.assert_called_with(self.new_share.id)
        self.assertCountEqual(self.columns, columns)
        self.assertCountEqual(self.datalist, data)