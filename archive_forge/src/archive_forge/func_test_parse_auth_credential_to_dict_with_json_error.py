import json
from unittest import mock
from keystoneauth1 import loading as ks_loading
from keystoneauth1 import session
from heat.common import auth_plugin
from heat.common import config
from heat.common import exception
from heat.common import policy
from heat.tests import common
def test_parse_auth_credential_to_dict_with_json_error(self):
    credential = "{'auth_type': v3applicationcredential, 'auth': {'auth_url': 'http://192.168.1.101/identity/v3', 'application_credential_id': '9dfa187e5a354484bf9c49a2b674333a', 'application_credential_secret': 'sec'} }"
    error = self.assertRaises(ValueError, auth_plugin.parse_auth_credential_to_dict, credential)
    error_msg = 'Failed to parse credential, please check your Stack Credential format.'
    self.assertEqual(error_msg, str(error))