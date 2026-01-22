from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data(True, False)
def test_list_share_networks_with_search_opts(self, with_search_opts):
    if with_search_opts:
        arglist = ['--name', 'foo', '--ip-version', '4', '--description~', 'foo-share-network']
        verifylist = [('name', 'foo'), ('ip_version', '4'), ('description~', 'foo-share-network')]
        self.expected_search_opts.update({'name': 'foo', 'ip_version': '4', 'description~': 'foo-share-network'})
    else:
        arglist = []
        verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
    self.assertEqual(COLUMNS, columns)
    self.assertEqual(list(self.values), list(data))