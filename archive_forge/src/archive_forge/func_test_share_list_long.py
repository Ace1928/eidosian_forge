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
def test_share_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True), ('all_projects', False)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    cmd_columns, cmd_data = self.cmd.take_action(parsed_args)
    search_opts = self._get_search_opts()
    self.shares_mock.list.assert_called_once_with(search_opts=search_opts)
    collist = ['ID', 'Name', 'Size', 'Share Protocol', 'Status', 'Is Public', 'Share Type Name', 'Availability Zone', 'Description', 'Share Network ID', 'Share Server ID', 'Share Type', 'Share Group ID', 'Host', 'User ID', 'Project ID', 'Access Rules Status', 'Source Snapshot ID', 'Supports Creating Snapshots', 'Supports Cloning Snapshots', 'Supports Mounting snapshots', 'Supports Reverting to Snapshot', 'Migration Task Status', 'Source Share Group Snapshot Member ID', 'Replication Type', 'Has Replicas', 'Created At', 'Properties']
    self.assertEqual(collist, cmd_columns)
    data = ((self.new_share.id, self.new_share.name, self.new_share.size, self.new_share.share_proto, self.new_share.status, self.new_share.is_public, self.new_share.share_type_name, self.new_share.availability_zone, self.new_share.description, self.new_share.share_network_id, self.new_share.share_server_id, self.new_share.share_type, self.new_share.share_group_id, self.new_share.host, self.new_share.user_id, self.new_share.project_id, self.new_share.access_rules_status, self.new_share.snapshot_id, self.new_share.snapshot_support, self.new_share.create_share_from_snapshot_support, self.new_share.mount_snapshot_support, self.new_share.revert_to_snapshot_support, self.new_share.task_state, self.new_share.source_share_group_snapshot_member_id, self.new_share.replication_type, self.new_share.has_replicas, self.new_share.created_at, self.new_share.metadata),)
    self.assertEqual(data, tuple(cmd_data))