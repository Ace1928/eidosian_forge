from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def run_and_validate(arglist, verifylist, attrs):
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.network_client.update_floating_ip_port_forwarding.assert_called_with(self._port_forwarding.floatingip_id, self._port_forwarding.id, **attrs)
    self.assertIsNone(result)