from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_group_list_by_project(self):
    self.epf_mock.list_endpoints_for_project.return_value = [self.endpoint_group]
    self.projects_mock.get.return_value = self.project
    arglist = ['--project', self.project.name, '--domain', self.domain.name]
    verifylist = [('project', self.project.name), ('domain', self.domain.name)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.epf_mock.list_endpoint_groups_for_project.assert_called_with(project=self.project.id)
    self.assertEqual(self.columns, columns)
    datalist = ((self.endpoint_group.id, self.endpoint_group.name, self.endpoint_group.description),)
    self.assertEqual(datalist, tuple(data))