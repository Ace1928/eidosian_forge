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
def test_create_port_with_extra_dhcp_option(self):
    extra_dhcp_options = [{'opt_name': 'classless-static-route', 'opt_value': '169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1', 'ip_version': '4'}, {'opt_name': 'dns-server', 'opt_value': '240C::6666', 'ip_version': '6'}]
    arglist = ['--network', self._port.network_id, '--extra-dhcp-option', 'name=classless-static-route,value=169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1,ip-version=4', '--extra-dhcp-option', 'name=dns-server,value=240C::6666,ip-version=6', 'test-port']
    verifylist = [('network', self._port.network_id), ('extra_dhcp_options', [{'name': 'classless-static-route', 'value': '169.254.169.254/32,22.2.0.2,0.0.0.0/0,22.2.0.1', 'ip-version': '4'}, {'name': 'dns-server', 'value': '240C::6666', 'ip-version': '6'}]), ('name', 'test-port')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'extra_dhcp_opts': extra_dhcp_options, 'name': 'test-port'})