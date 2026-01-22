from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_rule_create_default_rule(self, sgr_mock):
    expected_columns, expected_data = self._setup_security_group_rule()
    sgr_mock.return_value = self._security_group_rule
    dst_port = str(self._security_group_rule['from_port']) + ':' + str(self._security_group_rule['to_port'])
    arglist = ['--dst-port', dst_port, self._security_group['id']]
    verifylist = [('dst_port', (self._security_group_rule['from_port'], self._security_group_rule['to_port'])), ('group', self._security_group['id'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    sgr_mock.assert_called_once_with(security_group_id=self._security_group['id'], ip_protocol=self._security_group_rule['ip_protocol'], from_port=self._security_group_rule['from_port'], to_port=self._security_group_rule['to_port'], remote_ip=self._security_group_rule['ip_range']['cidr'], remote_group=None)
    self.assertEqual(expected_columns, columns)
    self.assertEqual(expected_data, data)