from unittest import mock
from keystoneauth1 import exceptions as ks_exc
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.identity.v2_0 import project
from openstackclient.tests.unit.identity.v2_0 import fakes as identity_fakes
@mock.patch.object(utils, 'find_resource')
def test_delete_multi_projects_with_exception(self, find_mock):
    find_mock.side_effect = [self.fake_project, exceptions.CommandError]
    arglist = [self.fake_project.id, 'unexist_project']
    verifylist = [('projects', arglist)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    try:
        self.cmd.take_action(parsed_args)
        self.fail('CommandError should be raised.')
    except exceptions.CommandError as e:
        self.assertEqual('1 of 2 projects failed to delete.', str(e))
    find_mock.assert_any_call(self.projects_mock, self.fake_project.id)
    find_mock.assert_any_call(self.projects_mock, 'unexist_project')
    self.assertEqual(2, find_mock.call_count)
    self.projects_mock.delete.assert_called_once_with(self.fake_project.id)