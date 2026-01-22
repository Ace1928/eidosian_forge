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
def test_retrieve_subject_token_session_error_idmsv2(self, utcnow):
    utcnow.return_value = datetime.datetime.strptime(self.AWS_SIGNATURE_TIME, '%Y-%m-%dT%H:%M:%SZ')
    request = self.make_mock_request(imdsv2_session_token_status=http_client.UNAUTHORIZED, imdsv2_session_token_data='unauthorized')
    credential_source_token_url = self.CREDENTIAL_SOURCE.copy()
    credential_source_token_url['imdsv2_session_token_url'] = IMDSV2_SESSION_TOKEN_URL
    credentials = self.make_credentials(credential_source=credential_source_token_url)
    with pytest.raises(exceptions.RefreshError) as excinfo:
        credentials.retrieve_subject_token(request)
    assert excinfo.match('Unable to retrieve AWS Session Token')
    self.assert_aws_metadata_request_kwargs(request.call_args_list[0][1], IMDSV2_SESSION_TOKEN_URL, {'X-aws-ec2-metadata-token-ttl-seconds': '300'}, 'PUT')