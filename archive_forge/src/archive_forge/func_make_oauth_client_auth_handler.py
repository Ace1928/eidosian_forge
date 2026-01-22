import json
import pytest  # type: ignore
from google.auth import exceptions
from google.oauth2 import utils
@classmethod
def make_oauth_client_auth_handler(cls, client_auth=None):
    return AuthHandler(client_auth)