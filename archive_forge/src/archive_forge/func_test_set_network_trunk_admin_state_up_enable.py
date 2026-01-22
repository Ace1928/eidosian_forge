import copy
from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
import testtools
from openstackclient.network.v2 import network_trunk
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as test_utils
def test_set_network_trunk_admin_state_up_enable(self):
    arglist = ['--enable', self._trunk['name']]
    verifylist = [('enable', True), ('trunk', self._trunk['name'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    attrs = {'admin_state_up': True}
    self.network_client.update_trunk.assert_called_once_with(self._trunk, **attrs)
    self.assertIsNone(result)