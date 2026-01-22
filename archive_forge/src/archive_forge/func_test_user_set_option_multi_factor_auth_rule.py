import contextlib
from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity import common
from openstackclient.identity.v3 import user
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_user_set_option_multi_factor_auth_rule(self):
    arglist = ['--multi-factor-auth-rule', identity_fakes.mfa_opt1, self.user.name]
    verifylist = [('name', None), ('password', None), ('email', None), ('multi_factor_auth_rule', [identity_fakes.mfa_opt1]), ('project', None), ('enable', False), ('disable', False), ('user', self.user.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': True, 'options': {'multi_factor_auth_rules': [['password', 'totp']]}}
    self.users_mock.update.assert_called_with(self.user.id, **kwargs)
    self.assertIsNone(result)