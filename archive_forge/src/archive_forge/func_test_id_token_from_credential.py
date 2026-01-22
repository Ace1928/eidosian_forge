import datetime
import json
import os
import mock
import pytest  # type: ignore
from six.moves import http_client
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import impersonated_credentials
from google.auth import transport
from google.auth.impersonated_credentials import Credentials
from google.oauth2 import credentials
from google.oauth2 import service_account
def test_id_token_from_credential(self, mock_donor_credentials, mock_authorizedsession_idtoken):
    credentials = self.make_credentials(lifetime=None)
    token = 'token'
    target_audience = 'https://foo.bar'
    expire_time = (_helpers.utcnow().replace(microsecond=0) + datetime.timedelta(seconds=500)).isoformat('T') + 'Z'
    response_body = {'accessToken': token, 'expireTime': expire_time}
    request = self.make_request(data=json.dumps(response_body), status=http_client.OK)
    credentials.refresh(request)
    assert credentials.valid
    assert not credentials.expired
    new_credentials = self.make_credentials(lifetime=None)
    id_creds = impersonated_credentials.IDTokenCredentials(credentials, target_audience=target_audience, include_email=True)
    id_creds = id_creds.from_credentials(target_credentials=new_credentials)
    id_creds.refresh(request)
    assert id_creds.token == ID_TOKEN_DATA
    assert id_creds._include_email is True
    assert id_creds._target_credentials is new_credentials