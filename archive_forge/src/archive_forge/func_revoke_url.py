import datetime
import io
import json
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.oauth2 import sts
from google.oauth2 import utils
@property
def revoke_url(self):
    """Optional[str]: The STS endpoint for token revocation."""
    return self._revoke_url