import datetime
import mock
import pytest  # type: ignore
from google.auth import app_engine
def test_service_account_email_explicit(self, app_identity):
    credentials = app_engine.Credentials(service_account_id=mock.sentinel.service_account_email)
    assert credentials.service_account_email == mock.sentinel.service_account_email
    assert not app_identity.get_service_account_name.called