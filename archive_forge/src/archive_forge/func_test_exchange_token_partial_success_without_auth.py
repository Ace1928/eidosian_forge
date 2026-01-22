import json
import mock
import pytest  # type: ignore
from six.moves import http_client
from six.moves import urllib
from google.auth import exceptions
from google.auth import transport
from google.oauth2 import sts
from google.oauth2 import utils
def test_exchange_token_partial_success_without_auth(self):
    """Test token exchange success without client authentication using
        partial (required only) parameters.
        """
    client = self.make_client()
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    request_data = {'grant_type': self.GRANT_TYPE, 'audience': self.AUDIENCE, 'requested_token_type': self.REQUESTED_TOKEN_TYPE, 'subject_token': self.SUBJECT_TOKEN, 'subject_token_type': self.SUBJECT_TOKEN_TYPE}
    request = self.make_mock_request(status=http_client.OK, data=self.SUCCESS_RESPONSE)
    response = client.exchange_token(request, grant_type=self.GRANT_TYPE, subject_token=self.SUBJECT_TOKEN, subject_token_type=self.SUBJECT_TOKEN_TYPE, audience=self.AUDIENCE, requested_token_type=self.REQUESTED_TOKEN_TYPE)
    self.assert_request_kwargs(request.call_args[1], headers, request_data)
    assert response == self.SUCCESS_RESPONSE