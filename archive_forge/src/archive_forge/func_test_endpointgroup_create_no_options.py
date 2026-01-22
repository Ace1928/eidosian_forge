from unittest import mock
from openstackclient.identity.v3 import endpoint_group
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_endpointgroup_create_no_options(self):
    arglist = ['--description', self.endpoint_group.description, self.endpoint_group.name, identity_fakes.endpoint_group_file_path]
    verifylist = [('name', self.endpoint_group.name), ('filters', identity_fakes.endpoint_group_file_path), ('description', self.endpoint_group.description)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    mocker = mock.Mock()
    mocker.return_value = identity_fakes.endpoint_group_filters
    with mock.patch('openstackclient.identity.v3.endpoint_group.CreateEndpointGroup._read_filters', mocker):
        columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'name': self.endpoint_group.name, 'filters': identity_fakes.endpoint_group_filters, 'description': self.endpoint_group.description}
    self.endpoint_groups_mock.create.assert_called_with(**kwargs)
    self.assertEqual(self.columns, columns)
    datalist = (self.endpoint_group.description, identity_fakes.endpoint_group_filters, self.endpoint_group.id, self.endpoint_group.name)
    self.assertEqual(datalist, data)