from unittest import mock
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_servers as osc_share_servers
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_server_list_by_status(self):
    arglist = ['--status', self.servers_list[0].status]
    verifylist = [('status', self.servers_list[0].status)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    search_opts = {'status': None, 'host': None, 'project_id': None}
    self.servers_mock.list.assert_called_once_with(search_opts=search_opts)
    self.assertEqual(self.columns, columns)
    self.assertEqual(list(self.values), list(data))