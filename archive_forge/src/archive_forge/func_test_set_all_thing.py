from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_all_thing(self):
    arglist_single = ['--port', self.port.id, '--internal-ip-address', 'new_internal_ip_address', '--internal-protocol-port', '100', '--external-protocol-port', '200', '--protocol', 'tcp', '--description', 'some description', self._port_forwarding.floatingip_id, self._port_forwarding.id]
    arglist_range = list(arglist_single)
    arglist_range[5] = '100:110'
    arglist_range[7] = '200:210'
    verifylist_single = [('port', self.port.id), ('internal_ip_address', 'new_internal_ip_address'), ('internal_protocol_port', '100'), ('external_protocol_port', '200'), ('protocol', 'tcp'), ('description', 'some description'), ('floating_ip', self._port_forwarding.floatingip_id), ('port_forwarding_id', self._port_forwarding.id)]
    verifylist_range = list(verifylist_single)
    verifylist_range[2] = ('internal_protocol_port', '100:110')
    verifylist_range[3] = ('external_protocol_port', '200:210')
    attrs_single = {'internal_port_id': self.port.id, 'internal_ip_address': 'new_internal_ip_address', 'internal_port': 100, 'external_port': 200, 'protocol': 'tcp', 'description': 'some description'}
    attrs_range = dict(attrs_single, internal_port_range='100:110', external_port_range='200:210')
    attrs_range.pop('internal_port')
    attrs_range.pop('external_port')

    def run_and_validate(arglist, verifylist, attrs):
        parsed_args = self.check_parser(self.cmd, arglist, verifylist)
        result = self.cmd.take_action(parsed_args)
        self.network_client.update_floating_ip_port_forwarding.assert_called_with(self._port_forwarding.floatingip_id, self._port_forwarding.id, **attrs)
        self.assertIsNone(result)
    run_and_validate(arglist_single, verifylist_single, attrs_single)
    run_and_validate(arglist_range, verifylist_range, attrs_range)