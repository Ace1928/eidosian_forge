from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
def test_project_list_long(self):
    arglist = ['--long']
    verifylist = [('long', True)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.projects_mock.list.assert_called_with()
    collist = ('ID', 'Name', 'Description', 'Enabled')
    self.assertEqual(collist, columns)
    datalist = ((self.fake_project.id, self.fake_project.name, self.fake_project.description, True),)
    self.assertEqual(datalist, tuple(data))