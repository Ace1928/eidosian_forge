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
def test_create_port_with_qos(self):
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
    arglist = ['--network', self._port.network_id, '--qos-policy', qos_policy.id, 'test-port']
    verifylist = [('network', self._port.network_id), ('enable', True), ('qos_policy', qos_policy.id), ('name', 'test-port')]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_port.assert_called_once_with(**{'admin_state_up': True, 'network_id': self._port.network_id, 'qos_policy_id': qos_policy.id, 'name': 'test-port'})
    self.assertCountEqual(self.columns, columns)
    self.assertCountEqual(self.data, data)