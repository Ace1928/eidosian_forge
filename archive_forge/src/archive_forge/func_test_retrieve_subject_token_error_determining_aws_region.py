import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import _helpers
from google.auth import aws
from google.auth import environment_vars
from google.auth import exceptions
from google.auth import transport
def test_retrieve_subject_token_error_determining_aws_region(self):
    request = self.make_mock_request(region_status=http_client.BAD_REQUEST)
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(request)
    assert excinfo.match('Unable to retrieve AWS region')