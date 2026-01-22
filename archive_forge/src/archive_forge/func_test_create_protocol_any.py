from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_protocol_any(self):
    self._setup_security_group_rule({'protocol': None, 'remote_ip_prefix': '10.0.2.0/24'})
    arglist = ['--proto', 'any', '--remote-ip', self._security_group_rule.remote_ip_prefix, self._security_group.id]
    verifylist = [('proto', 'any'), ('protocol', None), ('remote_ip', self._security_group_rule.remote_ip_prefix), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.create_security_group_rule.assert_called_once_with(**{'direction': self._security_group_rule.direction, 'ethertype': self._security_group_rule.ether_type, 'protocol': self._security_group_rule.protocol, 'remote_ip_prefix': self._security_group_rule.remote_ip_prefix, 'security_group_id': self._security_group.id})
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)