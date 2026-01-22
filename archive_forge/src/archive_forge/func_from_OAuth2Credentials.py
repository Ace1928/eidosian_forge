import datetime
import json
import logging
from oauth2client import _helpers
from oauth2client import client
from oauth2client import transport
from google_reauth import errors
from google_reauth import reauth
@classmethod
def from_OAuth2Credentials(cls, original):
    """Instantiate a Oauth2WithReauthCredentials from OAuth2Credentials."""
    json = original.to_json()
    return cls.from_json(json)