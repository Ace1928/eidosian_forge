from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_append_options(self):
    _testsubnet = network_fakes.FakeSubnet.create_one_subnet({'dns_nameservers': ['10.0.0.1'], 'service_types': ['network:router_gateway']})
    self.network_client.find_subnet = mock.Mock(return_value=_testsubnet)
    arglist = ['--dns-nameserver', '10.0.0.2', '--service-type', 'network:floatingip_agent_gateway', _testsubnet.name]
    verifylist = [('dns_nameservers', ['10.0.0.2']), ('service_types', ['network:floatingip_agent_gateway'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'dns_nameservers': ['10.0.0.2', '10.0.0.1'], 'service_types': ['network:floatingip_agent_gateway', 'network:router_gateway']}
    self.network_client.update_subnet.assert_called_once_with(_testsubnet, **attrs)
    self.assertIsNone(result)