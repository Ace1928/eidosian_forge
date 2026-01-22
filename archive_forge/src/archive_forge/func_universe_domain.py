import datetime
from google.auth import _helpers
from google.auth import credentials
from google.auth import exceptions
from google.auth import iam
from google.auth import jwt
from google.auth import metrics
from google.auth.compute_engine import _metadata
from google.auth.transport import requests as google_auth_requests
from google.oauth2 import _client
@property
def universe_domain(self):
    if self._universe_domain_cached:
        return self._universe_domain
    self._universe_domain = _metadata.get_universe_domain(self._universe_domain_request)
    self._universe_domain_cached = True
    return self._universe_domain