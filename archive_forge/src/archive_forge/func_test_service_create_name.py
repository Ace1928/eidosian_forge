from keystoneclient import exceptions as identity_exc
from osc_lib import exceptions
from openstackclient.identity.v3 import service
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
def test_service_create_name(self):
    arglist = ['--name', self.service.name, self.service.type]
    verifylist = [('name', self.service.name), ('description', None), ('enable', False), ('disable', False), ('type', self.service.type)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    self.services_mock.create.assert_called_with(name=self.service.name, type=self.service.type, description=None, enabled=True)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)