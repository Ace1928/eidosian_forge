import binascii
import hashlib
import hmac
import ipaddress
import logging
import urllib.parse as urlparse
import warnings
from oauthlib.common import extract_params, safe_string_equals, urldecode
from . import utils
def signature_base_string(http_method: str, base_str_uri: str, normalized_encoded_request_parameters: str) -> str:
    """
    Construct the signature base string.

    The *signature base string* is the value that is calculated and signed by
    the client. It is also independently calculated by the server to verify
    the signature, and therefore must produce the exact same value at both
    ends or the signature won't verify.

    The rules for calculating the *signature base string* are defined in
    section 3.4.1.1`_ of RFC 5849.

    .. _`section 3.4.1.1`: https://tools.ietf.org/html/rfc5849#section-3.4.1.1
    """
    base_string = utils.escape(http_method.upper())
    base_string += '&'
    base_string += utils.escape(base_str_uri)
    base_string += '&'
    base_string += utils.escape(normalized_encoded_request_parameters)
    return base_string