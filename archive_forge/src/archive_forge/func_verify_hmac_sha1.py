from __future__ import absolute_import, unicode_literals
import binascii
import hashlib
import hmac
import logging
from oauthlib.common import (extract_params, safe_string_equals, unicode_type,
from . import utils
def verify_hmac_sha1(request, client_secret=None, resource_owner_secret=None):
    """Verify a HMAC-SHA1 signature.

    Per `section 3.4`_ of the spec.

    .. _`section 3.4`: https://tools.ietf.org/html/rfc5849#section-3.4

    To satisfy `RFC2616 section 5.2`_ item 1, the request argument's uri
    attribute MUST be an absolute URI whose netloc part identifies the
    origin server or gateway on which the resource resides. Any Host
    item of the request argument's headers dict attribute will be
    ignored.

    .. _`RFC2616 section 5.2`: https://tools.ietf.org/html/rfc2616#section-5.2

    """
    norm_params = normalize_parameters(request.params)
    uri = normalize_base_string_uri(request.uri)
    base_string = construct_base_string(request.http_method, uri, norm_params)
    signature = sign_hmac_sha1(base_string, client_secret, resource_owner_secret)
    match = safe_string_equals(signature, request.signature)
    if not match:
        log.debug('Verify HMAC-SHA1 failed: sig base string: %s', base_string)
    return match