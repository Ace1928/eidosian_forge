import datetime
import pytest  # type: ignore
from google.auth import _credentials_async as credentials
from google.auth import _helpers
def test_expired_and_valid():
    credentials = CredentialsImpl()
    credentials.token = 'token'
    assert credentials.valid
    assert not credentials.expired
    credentials.expiry = datetime.datetime.utcnow() + _helpers.REFRESH_THRESHOLD + datetime.timedelta(seconds=1)
    assert credentials.valid
    assert not credentials.expired
    credentials.expiry = datetime.datetime.utcnow()
    assert not credentials.valid
    assert credentials.expired