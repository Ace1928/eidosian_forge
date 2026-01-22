import datetime
import json
import httplib2
from google.auth import aws
from google.auth import external_account
from google.auth import external_account_authorized_user
from google.auth import identity_pool
from google.auth import pluggable
from gslib.tests import testcase
from gslib.utils.wrapped_credentials import WrappedCredentials
import oauth2client
from six import add_move, MovedModule
from six.moves import mock
def testWrappedCredentialSerialization(self):
    """Test logic for converting Wrapped Credentials to and from JSON for serialization."""
    creds = WrappedCredentials(identity_pool.Credentials(audience='foo', subject_token_type='bar', token_url='https://sts.googleapis.com', credential_source={'url': 'google.com'}))
    creds.access_token = ACCESS_TOKEN
    creds.token_expiry = datetime.datetime(2001, 12, 5, 0, 0)
    creds_json = creds.to_json()
    json_values = json.loads(creds_json)
    self.assertEqual(json_values['client_id'], 'foo')
    self.assertEqual(json_values['access_token'], ACCESS_TOKEN)
    self.assertEqual(json_values['token_expiry'], '2001-12-05T00:00:00Z')
    self.assertEqual(json_values['_base']['audience'], 'foo')
    self.assertEqual(json_values['_base']['subject_token_type'], 'bar')
    self.assertEqual(json_values['_base']['token_url'], 'https://sts.googleapis.com')
    self.assertEqual(json_values['_base']['credential_source']['url'], 'google.com')
    creds2 = WrappedCredentials.from_json(creds_json)
    self.assertIsInstance(creds2, WrappedCredentials)
    self.assertIsInstance(creds2._base, identity_pool.Credentials)
    self.assertEqual(creds2.client_id, 'foo')
    self.assertEqual(creds2.access_token, ACCESS_TOKEN)
    self.assertEqual(creds2.token_expiry, creds.token_expiry)