import copy
from unittest import mock
from osc_lib import exceptions
from octaviaclient.osc.v2 import constants
from octaviaclient.osc.v2 import listener
from octaviaclient.tests.unit.osc.v2 import constants as attr_consts
from octaviaclient.tests.unit.osc.v2 import fakes
@mock.patch('octaviaclient.osc.v2.utils.get_listener_attrs')
def test_listener_create_timeouts(self, mock_client):
    mock_client.return_value = self.listener_info
    arglist = ['mock_lb_id', '--name', self._listener.name, '--protocol', 'HTTP', '--protocol-port', '80', '--timeout-client-data', '123', '--timeout-member-connect', '234', '--timeout-member-data', '345', '--timeout-tcp-inspect', '456']
    verifylist = [('loadbalancer', 'mock_lb_id'), ('name', self._listener.name), ('protocol', 'HTTP'), ('protocol_port', 80), ('timeout_client_data', 123), ('timeout_member_connect', 234), ('timeout_member_data', 345), ('timeout_tcp_inspect', 456)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.api_mock.listener_create.assert_called_with(json={'listener': self.listener_info})