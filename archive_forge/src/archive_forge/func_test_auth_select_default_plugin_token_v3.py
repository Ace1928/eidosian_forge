from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_select_default_plugin_token_v3(self):
    config = {'identity_api_version': '3', 'token': 'subway'}
    ret_config = self.cloud._auth_select_default_plugin(config)
    self.assertEqual('v3token', ret_config['auth_type'])
    self.assertEqual('subway', ret_config['token'])