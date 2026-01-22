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
def test_share_group_list_status(self):
    arglist = ['--status', self.new_share_group.status]
    verifylist = [('status', self.new_share_group.status)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    search_opts = {'all_tenants': False, 'name': None, 'status': None, 'share_server_id': None, 'share_group_type': None, 'snapshot': None, 'host': None, 'share_network': None, 'project_id': None, 'limit': None, 'offset': None, 'name~': None, 'description~': None, 'description': None}
    search_opts['status'] = self.new_share_group.status
    self.groups_mock.list.assert_called_once_with(search_opts=search_opts)
    self.assertEqual(self.column_headers, columns)
    self.assertEqual(list(self.values), list(data))