from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import l3_conntrack_helper
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_conntrack_helpers_list(self):
    arglist = [self.router.id]
    verifylist = [('router', self.router.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.conntrack_helpers.assert_called_once_with(self.router.id)
    self.assertEqual(self.columns, columns)
    list_data = list(data)
    self.assertEqual(len(self.data), len(list_data))
    for index in range(len(list_data)):
        self.assertEqual(self.data[index], list_data[index])