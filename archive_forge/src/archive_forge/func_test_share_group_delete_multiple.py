import argparse
from unittest import mock
import uuid
from osc_lib import exceptions
from osc_lib import exceptions as osc_exceptions
from osc_lib import utils as oscutils
from manilaclient.osc import utils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_groups as osc_share_groups
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_delete_multiple(self):
    share_groups = manila_fakes.FakeShareGroup.create_share_groups(count=2)
    arglist = [share_groups[0].id, share_groups[1].id]
    verifylist = [('share_group', [share_groups[0].id, share_groups[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.groups_mock.delete.call_count, len(share_groups))
    self.assertIsNone(result)