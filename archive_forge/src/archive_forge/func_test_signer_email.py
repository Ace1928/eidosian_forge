import datetime
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import _service_account_async as service_account
from tests.oauth2 import test_service_account
def test_signer_email(self):
    credentials = self.make_credentials()
    assert credentials.signer_email == self.SERVICE_ACCOUNT_EMAIL