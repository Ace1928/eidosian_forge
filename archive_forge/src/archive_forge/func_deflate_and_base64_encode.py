import base64
import hashlib
import hmac
import logging
import random
import string
import sys
import traceback
import zlib
from saml2 import VERSION
from saml2 import saml
from saml2 import samlp
from saml2.time_util import instant
def deflate_and_base64_encode(string_val):
    """
    Deflates and the base64 encodes a string

    :param string_val: The string to deflate and encode
    :return: The deflated and encoded string
    """
    if not isinstance(string_val, bytes):
        string_val = string_val.encode('utf-8')
    return base64.b64encode(zlib.compress(string_val)[2:-4])