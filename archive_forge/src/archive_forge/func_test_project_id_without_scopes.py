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
def test_project_id_without_scopes(self):
    credentials = CredentialsImpl(audience=self.AUDIENCE, subject_token_type=self.SUBJECT_TOKEN_TYPE, token_url=self.TOKEN_URL, credential_source=self.CREDENTIAL_SOURCE)
    assert credentials.get_project_id(None) is None