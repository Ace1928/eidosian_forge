from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
def get_token_from_header(request):
    """
    Helper function to extract a token from the request header.

    :param request: OAuthlib request.
    :type request: oauthlib.common.Request
    :return: Return the token or None if the Authorization header is malformed.
    """
    token = None
    if 'Authorization' in request.headers:
        split_header = request.headers.get('Authorization').split()
        if len(split_header) == 2 and split_header[0] == 'Bearer':
            token = split_header[1]
    else:
        token = request.access_token
    return token