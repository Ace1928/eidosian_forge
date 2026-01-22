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
@pytest.mark.parametrize('audience', TEST_USER_AUDIENCES)
def test_is_workforce_pool_with_users_and_impersonation(self, audience):
    credentials = CredentialsImpl(audience=audience, subject_token_type=self.SUBJECT_TOKEN_TYPE, token_url=self.TOKEN_URL, credential_source=self.CREDENTIAL_SOURCE, service_account_impersonation_url=self.SERVICE_ACCOUNT_IMPERSONATION_URL)
    assert credentials.is_workforce_pool is True