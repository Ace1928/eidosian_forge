from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import user
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_user_list_project(self):
    arglist = ['--project', self.fake_project_l.id]
    verifylist = [('project', self.fake_project_l.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    project_id = self.fake_project_l.id
    columns, data = self.cmd.take_action(parsed_args)
    self.users_mock.list.assert_called_with(tenant_id=project_id)
    self.assertEqual(self.columns, columns)
    self.assertCountEqual(self.datalist, tuple(data))