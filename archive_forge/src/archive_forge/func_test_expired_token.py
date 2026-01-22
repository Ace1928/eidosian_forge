import datetime
import json
import mock
import pytest  # type: ignore
from google.auth import _jwt_async as jwt_async
from google.auth import crypt
from google.auth import exceptions
from tests import test_jwt
def test_expired_token(self):
    self.credentials._cache['audience'] = (mock.sentinel.token, datetime.datetime.min)
    token = self.credentials._get_jwt_for_audience('audience')
    assert token != mock.sentinel.token