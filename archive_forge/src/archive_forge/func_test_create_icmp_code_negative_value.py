from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_create_icmp_code_negative_value(self):
    self._setup_security_group_rule({'port_range_min': 15, 'port_range_max': None, 'protocol': 'icmp', 'remote_ip_prefix': '0.0.0.0/0'})
    arglist = ['--protocol', self._security_group_rule.protocol, '--icmp-type', str(self._security_group_rule.port_range_min), '--icmp-code', '-2', self._security_group.id]
    verifylist = [('protocol', self._security_group_rule.protocol), ('icmp_type', self._security_group_rule.port_range_min), ('icmp_code', -2), ('group', self._security_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.expected_columns, columns)
    self.assertEqual(self.expected_data, data)