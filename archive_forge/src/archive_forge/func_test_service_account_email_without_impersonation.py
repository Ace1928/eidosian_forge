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
def test_service_account_email_without_impersonation(self):
    credentials = self.make_credentials()
    assert credentials.service_account_email is None