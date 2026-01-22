import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_delete_multiple(self):
    share_group_snapshots = manila_fakes.FakeShareGroupSnapshot.create_share_group_snapshots(count=2)
    arglist = [share_group_snapshots[0].id, share_group_snapshots[1].id]
    verifylist = [('share_group_snapshot', [share_group_snapshots[0].id, share_group_snapshots[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.group_snapshot_mocks.delete.call_count, len(share_group_snapshots))
    self.assertIsNone(result)