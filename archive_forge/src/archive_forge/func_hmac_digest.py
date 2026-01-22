import base64
from hashlib import sha256
import hmac
import binascii
from six import text_type, binary_type
def hmac_digest(key, data):
    return hmac.new(key, msg=data, digestmod=sha256).digest()