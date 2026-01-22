from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_list_sort(self):
    self.projects_mock.list.return_value = self.fake_projects
    arglist = ['--sort', 'name:asc']
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with()
    collist = ('ID', 'Name')
    self.assertEqual(collist, columns)
    if self.fake_projects[0].name > self.fake_projects[1].name:
        datalists = ((self.fake_projects[1].id, self.fake_projects[1].name), (self.fake_projects[0].id, self.fake_projects[0].name))
    else:
        datalists = ((self.fake_projects[0].id, self.fake_projects[0].name), (self.fake_projects[1].id, self.fake_projects[1].name))
    self.assertEqual(datalists, tuple(data))