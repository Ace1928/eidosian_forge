from datetime import datetime
import io
import json
import logging
import six
from google.auth import _cloud_sdk
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import reauth
@property
def granted_scopes(self):
    """Optional[Sequence[str]]: The OAuth 2.0 permission scopes that were granted by the user."""
    return self._granted_scopes