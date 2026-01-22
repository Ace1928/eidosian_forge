from unittest import mock
import mistralclient.tests.unit.base_shell_test as base
@mock.patch('mistralclient.api.client.client')
def test_with_domain_names_keystone_v3(self, client_mock):
    self.shell('--os-auth-url=https://127.0.0.1:35357/v3 --os-username=admin --os-password=1234 --os-project-domain-name=fake_domain --os-user-domain-name=fake_domain --os-target-project-domain-name=fake_domain --os-target-user-domain-name=fake_domain workbook-list')
    self.assertTrue(client_mock.called)
    params = client_mock.call_args
    self.assertEqual('https://127.0.0.1:35357/v3', params[1]['auth_url'])
    self.assertEqual('', params[1]['project_domain_id'])
    self.assertEqual('', params[1]['user_domain_id'])
    self.assertEqual('fake_domain', params[1]['project_domain_name'])
    self.assertEqual('fake_domain', params[1]['user_domain_name'])
    self.assertEqual('fake_domain', params[1]['target_project_domain_name'])
    self.assertEqual('fake_domain', params[1]['target_user_domain_name'])