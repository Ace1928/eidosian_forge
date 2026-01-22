from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnet_show(self):
    arglist = [self.share_network.id, self.share_network_subnet.id]
    verifylist = [('share_network', self.share_network.id), ('share_network_subnet', self.share_network_subnet.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_subnets_mock.get.assert_called_once_with(self.share_network.id, self.share_network_subnet.id)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)