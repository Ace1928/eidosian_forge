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
def test_refresh_with_retrieve_subject_token_error_url(self):
    credential_source = {'url': self.CREDENTIAL_URL, 'format': {'type': 'json', 'subject_token_field_name': 'not_found'}}
    credentials = self.make_credentials(credential_source=credential_source)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.refresh(self.make_mock_request(token_data=JSON_FILE_CONTENT))
    assert excinfo.match("Unable to parse subject_token from JSON file '{}' using key '{}'".format(self.CREDENTIAL_URL, 'not_found'))