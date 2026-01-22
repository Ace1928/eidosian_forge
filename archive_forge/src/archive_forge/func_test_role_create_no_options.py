from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import role
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_role_create_no_options(self):
    arglist = [self.fake_role_c.name]
    verifylist = [('role_name', self.fake_role_c.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.roles_mock.create.assert_called_with(self.fake_role_c.name)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)