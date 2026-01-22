from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.network.v2 import port
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_unset_port_parameters(self):
    arglist = ['--fixed-ip', 'subnet=042eb10a-3a18-4658-ab-cf47c8d03152,ip-address=1.0.0.0', '--binding-profile', 'Superman', '--qos-policy', '--host', self._testport.name]
    verifylist = [('fixed_ip', [{'subnet': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip-address': '1.0.0.0'}]), ('binding_profile', ['Superman']), ('qos_policy', True), ('host', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'fixed_ips': [{'subnet_id': '042eb10a-3a18-4658-ab-cf47c8d03152', 'ip_address': '0.0.0.1'}], 'binding:profile': {'batman': 'Joker'}, 'qos_policy_id': None, 'binding:host_id': None}
    self.network_client.update_port.assert_called_once_with(self._testport, **attrs)
    self.assertIsNone(result)