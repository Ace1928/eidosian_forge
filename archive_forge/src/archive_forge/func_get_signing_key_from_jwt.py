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
def get_signing_key_from_jwt(self, token: str) -> PyJWK:
    unverified = decode_token(token, options={'verify_signature': False})
    header = unverified['header']
    return self.get_signing_key(header.get('kid'))