from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
def test_set_share_network_status(self):
    arglist = [self.share_network.id, '--status', 'error']
    verifylist = [('share_network', self.share_network.id), ('status', 'error')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    with mock.patch('osc_lib.utils.find_resource', return_value=self.share_network):
        result = self.cmd.take_action(parsed_args)
    self.share_networks_mock.reset_state.assert_called_once_with(self.share_network, parsed_args.status)
    self.assertIsNone(result)