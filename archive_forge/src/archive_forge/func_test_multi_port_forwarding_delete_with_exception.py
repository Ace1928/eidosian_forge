from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip_port_forwarding
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_port_forwarding_delete_with_exception(self):
    arglist = [self.floating_ip.id, self._port_forwarding[0].id, 'unexist_port_forwarding_id']
    verifylist = [('floating_ip', self.floating_ip.id), ('port_forwarding_id', [self._port_forwarding[0].id, 'unexist_port_forwarding_id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    delete_mock_result = [None, exceptions.CommandError]
    self.network_client.delete_floating_ip_port_forwarding = mock.MagicMock(side_effect=delete_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 Port forwarding failed to delete.', str(e))
    self.network_client.delete_floating_ip_port_forwarding.assert_any_call(self.floating_ip.id, 'unexist_port_forwarding_id', ignore_missing=False)
    self.network_client.delete_floating_ip_port_forwarding.assert_any_call(self.floating_ip.id, self._port_forwarding[0].id, ignore_missing=False)