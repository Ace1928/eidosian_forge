from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def sign_plaintext(client_secret, resource_owner_secret):
    """Sign a request using plaintext.

    Per `section 3.4.4`_ of the spec.

    The "PLAINTEXT" method does not employ a signature algorithm.  It
    MUST be used with a transport-layer mechanism such as TLS or SSL (or
    sent over a secure channel with equivalent protections).  It does not
    utilize the signature base string or the "oauth_timestamp" and
    "oauth_nonce" parameters.

    .. _`section 3.4.4`: https://tools.ietf.org/html/rfc5849#section-3.4.4

    """
    signature = utils.escape(client_secret or '')
    signature += '&'
    signature += utils.escape(resource_owner_secret or '')
    return signature