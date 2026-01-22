import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.oauth2._client.call_iam_generate_id_token_endpoint', autospec=True)
def test_refresh_iam_flow(self, call_iam_generate_id_token_endpoint):
    credentials = self.make_credentials()
    credentials._use_iam_endpoint = True
    token = 'id_token'
    call_iam_generate_id_token_endpoint.return_value = (token, _helpers.utcnow() + datetime.timedelta(seconds=500))
    request = mock.Mock()
    credentials.refresh(request)
    req, signer_email, target_audience, access_token = call_iam_generate_id_token_endpoint.call_args[0]
    assert req == request
    assert signer_email == 'service-account@example.com'
    assert target_audience == 'https://example.com'
    decoded_access_token = jwt.decode(access_token, verify=False)
    assert decoded_access_token['scope'] == 'https://www.googleapis.com/auth/iam'