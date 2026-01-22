from __future__ import absolute_import, unicode_literals
import hashlib
import hmac
from binascii import b2a_base64
import warnings
from oauthlib import common
from oauthlib.common import add_params_to_qs, add_params_to_uri, unicode_type
from . import utils
def random_token_generator(request, refresh_token=False):
    """
    :param request: OAuthlib request.
    :type request: oauthlib.common.Request
    :param refresh_token:
    """
    return common.generate_token()