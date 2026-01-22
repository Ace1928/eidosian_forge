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
def test_set_port_security_group(self):
    sg = network_fakes.FakeSecurityGroup.create_one_security_group()
    self.network_client.find_security_group = mock.Mock(return_value=sg)
    arglist = ['--security-group', sg.id, self._port.name]
    verifylist = [('security_group', [sg.id]), ('port', self._port.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'security_group_ids': [sg.id]}
    self.network_client.update_port.assert_called_once_with(self._port, **attrs)
    self.assertIsNone(result)