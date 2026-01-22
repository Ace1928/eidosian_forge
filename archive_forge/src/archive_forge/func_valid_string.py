import base64
import calendar
from ipaddress import AddressValueError
from ipaddress import IPv4Address
from ipaddress import IPv6Address
import re
import struct
import time
from urllib.parse import urlparse
from saml2 import time_util
def valid_string(val):
    """Expects unicode
    Char ::= #x9 | #xA | #xD | [#x20-#xD7FF] | [#xE000-#xFFFD] |
                    [#x10000-#x10FFFF]
    """
    for char in val:
        try:
            char = ord(char)
        except TypeError:
            raise NotValid('string')
        if char == 9 or char == 10 or char == 13:
            continue
        elif 32 <= char <= 55295:
            continue
        elif 57344 <= char <= 65533:
            continue
        elif 65536 <= char <= 1114111:
            continue
        else:
            raise NotValid('string')
    return True