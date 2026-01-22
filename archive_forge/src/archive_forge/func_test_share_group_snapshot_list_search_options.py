import logging
import uuid
from osc_lib import exceptions
from osc_lib import utils as oscutils
from unittest import mock
from manilaclient import api_versions
from manilaclient.osc.v2 import (
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_group_snapshot_list_search_options(self):
    arglist = ['--name', self.share_group_snapshot.name, '--status', self.share_group_snapshot.status, '--share-group', self.share_group.id]
    verifylist = [('name', self.share_group_snapshot.name), ('status', self.share_group_snapshot.status), ('share_group', self.share_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.groups_mock.get.assert_called_with(self.share_group.id)
    self.group_snapshot_mocks.list.assert_called_once_with(search_opts={'all_tenants': False, 'name': self.share_group_snapshot.name, 'status': self.share_group_snapshot.status, 'share_group_id': self.share_group.id, 'limit': None, 'offset': None})
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))