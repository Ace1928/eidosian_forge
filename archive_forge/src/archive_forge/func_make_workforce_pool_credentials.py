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
@classmethod
def make_workforce_pool_credentials(cls, client_id=None, client_secret=None, quota_project_id=None, scopes=None, default_scopes=None, service_account_impersonation_url=None, workforce_pool_user_project=None):
    return CredentialsImpl(audience=cls.WORKFORCE_AUDIENCE, subject_token_type=cls.WORKFORCE_SUBJECT_TOKEN_TYPE, token_url=cls.TOKEN_URL, service_account_impersonation_url=service_account_impersonation_url, credential_source=cls.CREDENTIAL_SOURCE, client_id=client_id, client_secret=client_secret, quota_project_id=quota_project_id, scopes=scopes, default_scopes=default_scopes, workforce_pool_user_project=workforce_pool_user_project)