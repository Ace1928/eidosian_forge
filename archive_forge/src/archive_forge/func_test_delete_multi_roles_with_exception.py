from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
@mock.patch.object(utils, 'find_resource')
def test_delete_multi_roles_with_exception(self, find_mock):
    find_mock.side_effect = [self.fake_role, exceptions.CommandError]
    arglist = [self.fake_role.id, 'unexist_role']
    verifylist = [('roles', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 roles failed to delete.', str(e))
    find_mock.assert_any_call(self.roles_mock, self.fake_role.id)
    find_mock.assert_any_call(self.roles_mock, 'unexist_role')
    self.assertEqual(2, find_mock.call_count)
    self.roles_mock.delete.assert_called_once_with(self.fake_role.id)