import datetime
import json
import os
import mock
from google.auth import _helpers
from google.auth import crypt
from google.auth import jwt
from google.auth import transport
from google.oauth2 import service_account
def test__with_always_use_jwt_access(self):
    credentials = self.make_credentials()
    assert not credentials._always_use_jwt_access
    new_credentials = credentials.with_always_use_jwt_access(True)
    assert new_credentials._always_use_jwt_access