import base64
import binascii
import ipaddress
import json
import webbrowser
from datetime import datetime
import six
from pymacaroons import Macaroon
from pymacaroons.serializers import json_serializer
import six.moves.http_cookiejar as http_cookiejar
from six.moves.urllib.parse import urlparse
def b64decode(s):
    """Base64 decodes a base64-encoded string in URL-safe
    or normal format, with or without padding.
    The argument may be string or bytes.

    @param s bytes decode
    @return bytes decoded
    @raises ValueError on failure
    """
    s = to_bytes(s)
    if not s.endswith(b'='):
        s = s + b'=' * (-len(s) % 4)
    try:
        if '_' or '-' in s:
            return base64.urlsafe_b64decode(s)
        else:
            return base64.b64decode(s)
    except (TypeError, binascii.Error) as e:
        raise ValueError(str(e))