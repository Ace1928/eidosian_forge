import json
import urllib.request
from functools import lru_cache
from ssl import SSLContext
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache
def get_signing_key(self, kid: str) -> PyJWK:
    signing_keys = self.get_signing_keys()
    signing_key = self.match_kid(signing_keys, kid)
    if not signing_key:
        signing_keys = self.get_signing_keys(refresh=True)
        signing_key = self.match_kid(signing_keys, kid)
        if not signing_key:
            raise PyJWKClientError(f'Unable to find a signing key that matches: "{kid}"')
    return signing_key