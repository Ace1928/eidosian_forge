from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_group_list_projects_by_endpoint_group(self):
    arglist = ['--endpointgroup', self.endpoint_group.id]
    verifylist = [('endpointgroup', self.endpoint_group.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.epf_mock.list_projects_for_endpoint_group.assert_called_with(endpoint_group=self.endpoint_group.id)
    self.assertEqual(self.columns, columns)
    datalist = ((self.project.id, self.project.name, self.project.description),)
    self.assertEqual(datalist, tuple(data))