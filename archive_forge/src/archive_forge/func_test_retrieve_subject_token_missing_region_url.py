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
def test_retrieve_subject_token_missing_region_url(self):
    credential_source = self.CREDENTIAL_SOURCE.copy()
    credential_source.pop('region_url')
    credentials = self.make_credentials(credential_source=credential_source)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(None)
    assert excinfo.match('Unable to determine AWS region')