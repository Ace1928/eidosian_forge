from unittest import mock
from unittest.mock import call
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.network.v2 import subnet as subnet_v2
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_subnets_delete_with_exception(self):
    arglist = [self._subnets[0].name, 'unexist_subnet']
    verifylist = [('subnet', [self._subnets[0].name, 'unexist_subnet'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._subnets[0], exceptions.CommandError]
    self.network_client.find_subnet = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 subnets failed to delete.', str(e))
    self.network_client.find_subnet.assert_any_call(self._subnets[0].name, ignore_missing=False)
    self.network_client.find_subnet.assert_any_call('unexist_subnet', ignore_missing=False)
    self.network_client.delete_subnet.assert_called_once_with(self._subnets[0])