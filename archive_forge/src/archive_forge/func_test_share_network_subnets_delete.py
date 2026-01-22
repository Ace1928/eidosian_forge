from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from manilaclient import api_versions
from manilaclient.osc.v2 import share_network_subnets as osc_share_subnets
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_share_network_subnets_delete(self):
    arglist = [self.share_network.id, self.share_network_subnets[0].id, self.share_network_subnets[1].id]
    verifylist = [('share_network', self.share_network.id), ('share_network_subnet', [self.share_network_subnets[0].id, self.share_network_subnets[1].id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.assertEqual(self.share_subnets_mock.delete.call_count, len(self.share_network_subnets))
    self.assertIsNone(result)