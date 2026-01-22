import copy
import json
import time
import unittest
import uuid
from keystoneauth1 import _utils as ksa_utils
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1.exceptions import ClientException
from keystoneauth1 import fixture
from keystoneauth1.identity import v3
from keystoneauth1.identity.v3 import base as v3_base
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_oauth2_client_credential_method_500(self):
    self.TEST_URL = self.TEST_URL.replace('http:', 'https:')
    base_https = self.TEST_URL
    oauth2_endpoint = f'{base_https}/oauth_token'
    self.stub_auth(json=self.TEST_APP_CRED_TOKEN_RESPONSE)
    client_cre = v3.OAuth2ClientCredential(base_https, oauth2_endpoint=oauth2_endpoint, oauth2_client_id=self.TEST_CLIENT_CRED_ID, oauth2_client_secret=self.TEST_CLIENT_CRED_SECRET)
    oauth2_resp = {'status_code': 500, 'json': {'error': 'other_error', 'error_description': 'Unknown error is occur.'}}
    self.requests_mock.post(oauth2_endpoint, [oauth2_resp])
    sess = session.Session(auth=client_cre)
    err = self.assertRaises(ClientException, sess.get_auth_headers)
    self.assertEqual('Unknown error is occur.', str(err))