from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_target_user_name_and_password(self, client_mock):
    self.shell('--os-target-username=admin --os-target-password=secret_pass workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('admin', params[1]['target_username'])
    self.assertEqual('secret_pass', params[1]['target_api_key'])