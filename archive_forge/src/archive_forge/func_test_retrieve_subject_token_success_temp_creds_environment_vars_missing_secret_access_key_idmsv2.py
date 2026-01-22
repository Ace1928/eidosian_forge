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
@mock.patch.dict(os.environ, {environment_vars.AWS_REGION: AWS_REGION, environment_vars.AWS_ACCESS_KEY_ID: ACCESS_KEY_ID})
def test_retrieve_subject_token_success_temp_creds_environment_vars_missing_secret_access_key_idmsv2(self, utcnow):
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    request = self.make_mock_request(role_status=http_client.OK, role_name=self.AWS_ROLE, security_credentials_status=http_client.OK, security_credentials_data=self.AWS_SECURITY_CREDENTIALS_RESPONSE, imdsv2_session_token_status=http_client.OK, imdsv2_session_token_data=self.AWS_IMDSV2_SESSION_TOKEN)
    credential_source_token_url = self.CREDENTIAL_SOURCE.copy()
    credential_source_token_url['imdsv2_session_token_url'] = IMDSV2_SESSION_TOKEN_URL
    credentials = self.make_credentials(credential_source=credential_source_token_url)
    subject_token = credentials.retrieve_subject_token(request)
    assert subject_token == self.make_serialized_aws_signed_request({'access_key_id': ACCESS_KEY_ID, 'secret_access_key': SECRET_ACCESS_KEY, 'security_token': TOKEN})
    self.assert_aws_metadata_request_kwargs(request.call_args_list[0][1], IMDSV2_SESSION_TOKEN_URL, {'X-aws-ec2-metadata-token-ttl-seconds': '300'}, 'PUT')
    self.assert_aws_metadata_request_kwargs(request.call_args_list[1][1], SECURITY_CREDS_URL, {'X-aws-ec2-metadata-token': self.AWS_IMDSV2_SESSION_TOKEN})
    self.assert_aws_metadata_request_kwargs(request.call_args_list[2][1], '{}/{}'.format(SECURITY_CREDS_URL, self.AWS_ROLE), {'Content-Type': 'application/json', 'X-aws-ec2-metadata-token': self.AWS_IMDSV2_SESSION_TOKEN})