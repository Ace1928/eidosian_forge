import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test_from_service_account_file_args(self):
    info = test_service_account.SERVICE_ACCOUNT_INFO.copy()
    scopes = ['email', 'profile']
    subject = 'subject'
    additional_claims = {'meta': 'data'}
    credentials = service_account.Credentials.from_service_account_file(test_service_account.SERVICE_ACCOUNT_JSON_FILE, subject=subject, scopes=scopes, additional_claims=additional_claims)
    assert credentials.service_account_email == info['client_email']
    assert credentials.project_id == info['project_id']
    assert credentials._signer.key_id == info['private_key_id']
    assert credentials._token_uri == info['token_uri']
    assert credentials._scopes == scopes
    assert credentials._subject == subject
    assert credentials._additional_claims == additional_claims