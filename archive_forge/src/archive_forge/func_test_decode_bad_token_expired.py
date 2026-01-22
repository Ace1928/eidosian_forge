import base64
import datetime
import json
import os
import mock
import pytest  # type: ignore
from google.auth import _helpers
from google.auth import crypt
from google.auth import exceptions
from google.auth import jwt
def test_decode_bad_token_expired(token_factory):
    token = token_factory(claims={'exp': _helpers.datetime_to_secs(_helpers.utcnow() - datetime.timedelta(hours=1))})
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token, PUBLIC_CERT_BYTES, clock_skew_in_seconds=59)
    assert excinfo.match('Token expired')