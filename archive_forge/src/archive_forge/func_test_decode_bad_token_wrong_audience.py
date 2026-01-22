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
def test_decode_bad_token_wrong_audience(token_factory):
    token = token_factory()
    audience = 'audience2@example.com'
    with pytest.raises(ValueError) as excinfo:
        jwt.decode(token, PUBLIC_CERT_BYTES, audience=audience)
    assert excinfo.match('Token has wrong audience')