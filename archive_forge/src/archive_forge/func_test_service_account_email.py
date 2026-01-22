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
def test_service_account_email(self):
    credentials = self.make_credentials(target_principal=self.TARGET_PRINCIPAL)
    assert credentials.service_account_email == self.TARGET_PRINCIPAL