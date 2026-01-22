from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_create_password_prompt(self):
    arglist = ['--password-prompt', self.fake_user_c.name]
    verifylist = [('name', self.fake_user_c.name), ('password_prompt', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = 'abc123'
    with mock.patch('osc_lib.utils.get_password', mocker):
        columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': True, 'tenant_id': None}
    self.users_mock.create.assert_called_with(self.fake_user_c.name, 'abc123', None, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)