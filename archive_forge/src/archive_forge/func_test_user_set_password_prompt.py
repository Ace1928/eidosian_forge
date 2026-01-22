from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_set_password_prompt(self):
    arglist = ['--password-prompt', self.fake_user.name]
    verifylist = [('name', None), ('password', None), ('password_prompt', True), ('email', None), ('project', None), ('enable', False), ('disable', False), ('user', self.fake_user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = 'abc123'
    with mock.patch('osc_lib.utils.get_password', mocker):
        result = self.cmd.take_action(parsed_args)
    self.users_mock.update_password.assert_called_with(self.fake_user.id, 'abc123')
    self.assertIsNone(result)