import random
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import network
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_network_list_enable(self):
    arglist = ['--enable']
    verifylist = [('long', False), ('external', False), ('enable', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.network_client.networks.assert_called_once_with(**{'admin_state_up': True, 'is_admin_state_up': True})
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.data, list(data))