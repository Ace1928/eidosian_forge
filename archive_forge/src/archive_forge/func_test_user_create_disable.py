from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_create_disable(self):
    arglist = ['--disable', self.fake_user_c.name]
    verifylist = [('name', self.fake_user_c.name), ('enable', False), ('disable', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'enabled': False, 'tenant_id': None}
    self.users_mock.create.assert_called_with(self.fake_user_c.name, None, None, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)