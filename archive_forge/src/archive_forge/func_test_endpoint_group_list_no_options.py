from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpoint_group_list_no_options(self):
    arglist = []
    verifylist = []
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.endpoint_groups_mock.list.assert_called_with()
    self.assertEqual(self.columns, columns)
    datalist = ((self.endpoint_group.id, self.endpoint_group.name, self.endpoint_group.description),)
    self.assertEqual(datalist, tuple(data))