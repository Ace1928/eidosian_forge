import requests_mock
from castellan.key_manager import vault_key_manager
from castellan.tests.unit.key_manager import test_key_manager
def test_auth_headers_root_token_with_namespace(self):
    self.key_mgr._root_token_id = 'spam'
    self.key_mgr._namespace = 'ham'
    expected_headers = {'X-Vault-Token': 'spam', 'X-Vault-Namespace': 'ham'}
    self.assertEqual(expected_headers, self.key_mgr._build_auth_headers())