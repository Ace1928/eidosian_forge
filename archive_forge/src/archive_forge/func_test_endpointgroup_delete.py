from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpointgroup_delete(self):
    arglist = [self.endpoint_group.id]
    verifylist = [('endpointgroup', [self.endpoint_group.id])]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    result = self.cmd.take_action(parsed_args)
    self.endpoint_groups_mock.delete.assert_called_with(self.endpoint_group.id)
    self.assertIsNone(result)