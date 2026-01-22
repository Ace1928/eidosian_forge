from unittest import mock
from osc_lib import exceptions
from openstackclient.network.v2 import network_flavor
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes_v3
from openstackclient.tests.unit.network.v2 import fakes as network_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_multi_network_flavors_delete_with_exception(self):
    arglist = [self._network_flavors[0].name, 'unexist_network_flavor']
    verifylist = [('flavor', [self._network_flavors[0].name, 'unexist_network_flavor'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    find_mock_result = [self._network_flavors[0], exceptions.CommandError]
    self.network_client.find_flavor = mock.Mock(side_effect=find_mock_result)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 flavors failed to delete.', str(e))
    self.network_client.find_flavor.assert_any_call(self._network_flavors[0].name, ignore_missing=False)
    self.network_client.find_flavor.assert_any_call('unexist_network_flavor', ignore_missing=False)
    self.network_client.delete_flavor.assert_called_once_with(self._network_flavors[0])