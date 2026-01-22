from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_endpoint_type(self, client_mock):
    self.shell('--os-mistral-endpoint-type=adminURL workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('adminURL', params[1]['endpoint_type'])