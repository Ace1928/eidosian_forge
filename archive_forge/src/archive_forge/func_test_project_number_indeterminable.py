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
@pytest.mark.parametrize('audience', ['identitynamespace:1f12345:my_provider', '//iam.googleapis.com/projects', '//iam.googleapis.com/projects/', '//iam.googleapis.com/project/123456', '//iam.googleapis.com/projects//123456', '//iam.googleapis.com/prefix_projects/123456', '//iam.googleapis.com/projects_suffix/123456'])
def test_project_number_indeterminable(self, audience):
    credentials = CredentialsImpl(audience=audience, subject_token_type=self.SUBJECT_TOKEN_TYPE, token_url=self.TOKEN_URL, credential_source=self.CREDENTIAL_SOURCE)
    assert credentials.project_number is None
    assert credentials.get_project_id(None) is None