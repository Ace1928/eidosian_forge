from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_command_interactive_mode(self, client_mock):
    self.shell('')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('', params[1]['mistral_url'])