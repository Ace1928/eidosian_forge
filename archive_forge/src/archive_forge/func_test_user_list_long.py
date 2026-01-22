from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.users_mock.list.assert_called_with(tenant_id=None)
    collist = ('ID', 'Name', 'Project', 'Email', 'Enabled')
    self.assertEqual(collist, columns)
    datalist = ((self.fake_user_l.id, self.fake_user_l.name, user.ProjectColumn(self.fake_project_l.id, {self.fake_project_l.id: self.fake_project_l}), self.fake_user_l.email, True),)
    self.assertCountEqual(datalist, tuple(data))