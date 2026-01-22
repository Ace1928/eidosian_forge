from urllib.parse import urlparse
import logging
from oauthlib.common import add_params_to_uri
from oauthlib.common import urldecode as _urldecode
from oauthlib.oauth1 import SIGNATURE_HMAC, SIGNATURE_RSA, SIGNATURE_TYPE_AUTH_HEADER
import requests
from . import OAuth1
def rebuild_auth(self, prepared_request, response):
    """
        When being redirected we should always strip Authorization
        header, since nonce may not be reused as per OAuth spec.
        """
    if 'Authorization' in prepared_request.headers:
        prepared_request.headers.pop('Authorization', True)
        prepared_request.prepare_auth(self.auth)
    return