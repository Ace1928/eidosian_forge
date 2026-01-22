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
@mock.patch('google.auth._helpers.utcnow')
def test_refresh_success_without_impersonation_use_default_scopes(self, utcnow):
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    expected_subject_token = self.make_serialized_aws_signed_request({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY, 'security_token': TOKEN})
    token_headers = {'Content-Type': 'application/x-www-form-urlencoded', 'Authorization': 'Basic ' + BASIC_AUTH_ENCODING}
    token_request_data = {'grant_type': 'urn:ietf:params:oauth:grant-type:token-exchange', 'audience': AUDIENCE, 'requested_token_type': 'urn:ietf:params:oauth:token-type:access_token', 'scope': ' '.join(SCOPES), 'subject_token': expected_subject_token, 'subject_token_type': SUBJECT_TOKEN_TYPE}
    request = self.make_mock_request(region_status=http_client.OK, region_name=self.AWS_REGION, role_status=http_client.OK, role_name=self.AWS_ROLE, security_credentials_status=http_client.OK, security_credentials_data=self.AWS_SECURITY_CREDENTIALS_RESPONSE, token_status=http_client.OK, token_data=self.SUCCESS_RESPONSE)
    credentials = self.make_credentials(client_id=CLIENT_ID, client_secret=CLIENT_SECRET, credential_source=self.CREDENTIAL_SOURCE, quota_project_id=QUOTA_PROJECT_ID, scopes=None, default_scopes=SCOPES)
    credentials.refresh(request)
    assert len(request.call_args_list) == 4
    self.assert_token_request_kwargs(request.call_args_list[3][1], token_headers, token_request_data)
    assert credentials.token == self.SUCCESS_RESPONSE['access_token']
    assert credentials.quota_project_id == QUOTA_PROJECT_ID
    assert credentials.scopes is None
    assert credentials.default_scopes == SCOPES