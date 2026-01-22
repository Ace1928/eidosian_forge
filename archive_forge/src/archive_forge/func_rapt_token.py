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
def rapt_token(self):
    """Optional[str]: The reauth Proof Token."""
    return self._rapt_token