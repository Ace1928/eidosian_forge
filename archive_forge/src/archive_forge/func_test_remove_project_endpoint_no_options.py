from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_remove_project_endpoint_no_options(self):
    arglist = [self.endpoint_group.id, self.project.id]
    verifylist = [('endpointgroup', self.endpoint_group.id), ('project', self.project.id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.epf_mock.delete_endpoint_group_from_project.assert_called_with(project=self.project.id, endpoint_group=self.endpoint_group.id)
    self.assertIsNone(result)