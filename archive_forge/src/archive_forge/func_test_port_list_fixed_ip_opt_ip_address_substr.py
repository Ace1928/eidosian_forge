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
def test_port_list_fixed_ip_opt_ip_address_substr(self):
    ip_address_ss = self._ports[0].fixed_ips[0]['ip_address'][:-1]
    arglist = ['--fixed-ip', 'ip-substring=%s' % ip_address_ss]
    verifylist = [('fixed_ip', [{'ip-substring': ip_address_ss}])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.ports.assert_called_once_with(**{'fixed_ips': ['ip_address_substr=%s' % ip_address_ss], 'fields': LIST_FIELDS_TO_RETRIEVE})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))