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
def test_info_workforce_pool(self):
    credentials = self.make_workforce_pool_credentials(workforce_pool_user_project=self.WORKFORCE_POOL_USER_PROJECT)
    assert credentials.info == {'type': 'external_account', 'audience': self.WORKFORCE_AUDIENCE, 'subject_token_type': self.WORKFORCE_SUBJECT_TOKEN_TYPE, 'token_url': self.TOKEN_URL, 'credential_source': self.CREDENTIAL_SOURCE.copy(), 'workforce_pool_user_project': self.WORKFORCE_POOL_USER_PROJECT}