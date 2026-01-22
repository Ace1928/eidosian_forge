import datetime
import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import exceptions
from google.auth import external_account
from google.auth import transport
def test_with_scopes_full_options_propagated(self):
    credentials = self.make_credentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, quota_project_id=self.QUOTA_PROJECT_ID, scopes=self.SCOPES, token_info_url=self.TOKEN_INFO_URL, default_scopes=['default1'], service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, service_account_impersonation_options={'token_lifetime_seconds': 2800})
    with mock.patch.object(external_account.Credentials, '__init__', return_value=None) as mock_init:
        credentials.with_scopes(['email'], ['default2'])
    mock_init.assert_called_once_with(audience=self.AUDIENCE, subject_token_type=self.SUBJECT_TOKEN_TYPE, token_url=self.TOKEN_URL, token_info_url=self.TOKEN_INFO_URL, credential_source=self.CREDENTIAL_SOURCE, service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL, service_account_impersonation_options={'token_lifetime_seconds': 2800}, client_id=CLIENT_ID, client_secret=CLIENT_SECRET, quota_project_id=self.QUOTA_PROJECT_ID, scopes=['email'], default_scopes=['default2'])