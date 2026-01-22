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
def test_retrieve_subject_token_success_permanent_creds_no_environment_vars(self, utcnow):
    security_creds_response = self.AWS_SECURITY_CREDENTIALS_RESPONSE.copy()
    security_creds_response.pop('Token')
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    request = self.make_mock_request(region_status=http_client.OK, region_name=self.AWS_REGION, role_status=http_client.OK, role_name=self.AWS_ROLE, security_credentials_status=http_client.OK, security_credentials_data=security_creds_response)
    credentials = self.make_credentials(credential_source=self.CREDENTIAL_SOURCE)
    subject_token = credentials.retrieve_subject_token(request)
    assert subject_token == self.make_serialized_aws_signed_request({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY})