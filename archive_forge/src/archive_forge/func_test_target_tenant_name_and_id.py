from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_target_tenant_name_and_id(self, client_mock):
    self.shell('--os-target-tenant-id=123fake --os-target-tenant-name=fake_target workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('123fake', params[1]['target_project_id'])
    self.assertEqual('fake_target', params[1]['target_project_name'])