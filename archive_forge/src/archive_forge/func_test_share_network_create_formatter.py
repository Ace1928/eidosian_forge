from unittest import mock
import uuid
import ddt
from osc_lib import exceptions
from osc_lib import utils as oscutils
from manilaclient import api_versions
from manilaclient.osc.v2 import share_networks as osc_share_networks
from manilaclient.tests.unit.osc import osc_utils
from manilaclient.tests.unit.osc.v2 import fakes as manila_fakes
@ddt.data('table', 'yaml')
def test_share_network_create_formatter(self, formatter):
    arglist = ['-f', formatter]
    verifylist = [('formatter', formatter)]
    expected_data = self.share_network._info
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.share_networks_mock.create.assert_called_once_with(name=None, description=None, neutron_net_id=None, neutron_subnet_id=None, availability_zone=None)
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(expected_data.values(), data)