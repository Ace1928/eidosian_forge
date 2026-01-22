import copy
from openstackclient.identity.v3 import service_provider
from openstackclient.tests.unit import fakes
from openstackclient.tests.unit.identity.v3 import fakes as service_fakes
def test_create_service_provider_description(self):
    arglist = ['--description', service_fakes.sp_description, '--auth-url', service_fakes.sp_auth_url, '--service-provider-url', service_fakes.service_provider_url, service_fakes.sp_id]
    verifylist = [('description', service_fakes.sp_description), ('auth_url', service_fakes.sp_auth_url), ('service_provider_url', service_fakes.service_provider_url), ('service_provider_id', service_fakes.sp_id)]
    parsed_args = self.check_parser(self.cmd, arglist, verifylist)
    columns, data = self.cmd.take_action(parsed_args)
    kwargs = {'description': service_fakes.sp_description, 'auth_url': service_fakes.sp_auth_url, 'sp_url': service_fakes.service_provider_url, 'enabled': True}
    self.service_providers_mock.create.assert_called_with(id=service_fakes.sp_id, **kwargs)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.datalist, data)