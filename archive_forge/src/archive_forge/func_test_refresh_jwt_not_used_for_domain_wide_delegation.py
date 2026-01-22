import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
@mock.patch('google.oauth2._client.jwt_grant', autospec=True)
@mock.patch('google.auth.jwt.Credentials.refresh', autospec=True)
def test_refresh_jwt_not_used_for_domain_wide_delegation(self, self_signed_jwt_refresh, jwt_grant):
    credentials = service_account.Credentials(SIGNER, self.SERVICE_ACCOUNT_EMAIL, self.TOKEN_URI, always_use_jwt_access=True, subject='subject')
    credentials._create_self_signed_jwt('https://pubsub.googleapis.com')
    jwt_grant.return_value = ('token', _helpers.utcnow() + datetime.timedelta(seconds=500), {})
    request = mock.create_autospec(transport.Request, instance=True)
    credentials.refresh(request)
    assert jwt_grant.called
    assert not self_signed_jwt_refresh.called