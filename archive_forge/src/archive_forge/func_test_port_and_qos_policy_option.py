from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import floating_ip as fip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_port_and_qos_policy_option(self):
    qos_policy = network_fakes.FakeNetworkQosPolicy.create_one_qos_policy()
    self.network_client.find_qos_policy = mock.Mock(return_value=qos_policy)
    arglist = ['--qos-policy', qos_policy.id, '--port', self.floating_ip.port_id, self.floating_ip.id]
    verifylist = [('qos_policy', qos_policy.id), ('port', self.floating_ip.port_id), ('floating_ip', self.floating_ip.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.cmd.take_action(parsed_args)
    attrs = {'qos_policy_id': qos_policy.id, 'port_id': self.floating_ip.port_id}
    self.network_client.find_ip.assert_called_once_with(self.floating_ip.id, ignore_missing=False)
    self.network_client.update_ip.assert_called_once_with(self.floating_ip, **attrs)