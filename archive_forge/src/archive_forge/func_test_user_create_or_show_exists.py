from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_create_or_show_exists(self):

    def _raise_conflict(*args, **kwargs):
        raise ks_exc.Conflict(None)
    self.users_mock.create.side_effect = _raise_conflict
    self.users_mock.get.return_value = self.fake_user_c
    arglist = ['--or-show', self.fake_user_c.name]
    verifylist = [('name', self.fake_user_c.name), ('or_show', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.users_mock.get.assert_called_with(self.fake_user_c.name)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)