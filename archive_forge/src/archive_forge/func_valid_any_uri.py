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
def valid_any_uri(item):
    """very simplistic, ..."""
    try:
        part = urlparse(item)
    except Exception:
        raise NotValid('AnyURI')
    if part[0] == 'urn' and part[1] == '':
        return True
    return True