from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_project_domain_name(self, client_mock):
    self.shell('--os-project-domain-name=default workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('default', params[1]['project_domain_name'])