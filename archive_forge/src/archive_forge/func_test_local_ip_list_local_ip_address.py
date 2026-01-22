from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import local_ip
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_local_ip_list_local_ip_address(self):
    arglist = ['--local-ip-address', self.local_ips[0].local_ip_address]
    verifylist = [('local_ip_address', self.local_ips[0].local_ip_address)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.local_ips.assert_called_once_with(**{'local_ip_address': self.local_ips[0].local_ip_address})
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.data, list(data))