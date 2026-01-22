import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def raw_urlsafe_b64decode(s):
    """Base64 decode with added padding and conversion to bytes.

    @param s string decode
    @return bytes decoded
    """
    return base64.urlsafe_b64decode(add_base64_padding(s.encode('utf-8')))