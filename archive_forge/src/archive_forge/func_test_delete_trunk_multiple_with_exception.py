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
def test_delete_trunk_multiple_with_exception(self):
    arglist = [self.new_trunks[0].name, 'unexist_trunk']
    verifylist = [('trunk', [self.new_trunks[0].name, 'unexist_trunk'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    self.network_client.find_trunk = mock.Mock(side_effect=[self.new_trunks[0], exceptions.CommandError])
    with testtools.ExpectedException(exceptions.CommandError) as e:
        self.cmd.take_action(parsed_args)
        self.assertEqual('1 of 2 trunks failed to delete.', str(e))
    self.network_client.delete_trunk.assert_called_once_with(self.new_trunks[0].id)