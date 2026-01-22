import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import identity_pool
from google.auth import transport
def test_refresh_workforce_success_with_client_auth_and_no_workforce_project(self):
    credentials = self.make_credentials(audience=WORKFORCE_AUDIENCE, subject_token_type=WORKFORCE_SUBJECT_TOKEN_TYPE, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, credential_source=self.CREDENTIAL_SOURCE_TEXT, scopes=SCOPES, workforce_pool_user_project=None)
    self.assert_underlying_credentials_refresh(credentials=credentials, audience=WORKFORCE_AUDIENCE, subject_token=TEXT_FILE_SUBJECT_TOKEN, subject_token_type=WORKFORCE_SUBJECT_TOKEN_TYPE, token_url=TOKEN_URL, service_account_impersonation_url=None, basic_auth_encoding=BASIC_AUTH_ENCODING, quota_project_id=None, used_scopes=SCOPES, scopes=SCOPES, workforce_pool_user_project=None)