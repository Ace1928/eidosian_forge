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
@pytest.fixture
def es256_signer():
    return crypt.ES256Signer.from_string(EC_PRIVATE_KEY_BYTES, '1')