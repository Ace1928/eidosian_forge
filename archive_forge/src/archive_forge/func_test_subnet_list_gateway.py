from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_subnet_list_gateway(self):
    subnet = network_fakes.FakeSubnet.create_one_subnet()
    self.network_client.find_network = mock.Mock(return_value=subnet)
    arglist = ['--gateway', subnet.gateway_ip]
    verifylist = [('gateway', subnet.gateway_ip)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    filters = {'gateway_ip': subnet.gateway_ip}
    self.network_client.subnets.assert_called_once_with(**filters)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))