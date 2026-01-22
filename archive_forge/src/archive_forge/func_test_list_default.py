from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network import utils as network_utils
from openstackclient.network.v2 import security_group_rule
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_list_default(self):
    self._security_group_rule_tcp.port_range_min = 80
    parsed_args = self.check_parser(self.cmd, [], [])
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.security_group_rules.assert_called_once_with(**{})
    self.assertEqual(self.expected_columns_no_group, columns)
    self.assertEqual(self.expected_data_no_group, list(data))