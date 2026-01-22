from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import address_scope
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_address_scopes_delete_with_exception(self):
    arglist = [self._address_scopes[0].name, 'unexist_address_scope']
    verifylist = [('address_scope', [self._address_scopes[0].name, 'unexist_address_scope'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._address_scopes[0], exceptions.CommandError]
    self.network_client.find_address_scope = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 address scopes failed to delete.', str(e))
    self.network_client.find_address_scope.assert_any_call(self._address_scopes[0].name, ignore_missing=False)
    self.network_client.find_address_scope.assert_any_call('unexist_address_scope', ignore_missing=False)
    self.network_client.delete_address_scope.assert_called_once_with(self._address_scopes[0])