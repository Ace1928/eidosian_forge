from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import network_qos_policy
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_set_name_share_description_default(self):
    arglist = ['--name', 'new_qos_policy', '--share', '--description', 'QoS policy description', '--default', self._qos_policy.name]
    verifylist = [('name', 'new_qos_policy'), ('share', True), ('description', 'QoS policy description'), ('default', True), ('policy', self._qos_policy.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'name': 'new_qos_policy', 'description': 'QoS policy description', 'shared': True, 'is_default': True}
    self.network_client.update_qos_policy.assert_called_with(self._qos_policy, **attrs)
    self.assertIsNone(result)