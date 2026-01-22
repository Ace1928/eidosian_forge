from unittest import mock
from unittest.mock import call
from osc_lib import exceptions
from openstackclient.network.v2 import security_group
from openstackclient.tests.unit.compute.v2 import fakes as compute_fakes
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
def test_security_group_multi_delete_with_exception(self, sg_mock):
    sg_mock.return_value = mock.Mock(return_value=None)
    sg_mock.side_effect = [mock.Mock(return_value=None), exceptions.CommandError]
    arglist = [self._security_groups[0]['id'], 'unexist_security_group']
    verifylist = [('group', [self._security_groups[0]['id'], 'unexist_security_group'])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 groups failed to delete.', str(e))
    sg_mock.assert_any_call(self._security_groups[0]['id'])
    sg_mock.assert_any_call('unexist_security_group')