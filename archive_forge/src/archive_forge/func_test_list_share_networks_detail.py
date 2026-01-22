from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_list_share_networks_detail(self):
    values = (oscutils.get_dict_properties(s._info, COLUMNS_DETAIL) for s in self.share_networks_list)
    arglist = ['--detail']
    verifylist = [('detail', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_networks_mock.list.assert_called_once_with(search_opts=self.expected_search_opts)
    self.assertEqual(COLUMNS_DETAIL, columns)
    self.assertEqual(list(values), list(data))