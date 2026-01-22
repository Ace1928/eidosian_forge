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
def valid_non_negative_integer(val):
    try:
        integer = int(val)
    except ValueError:
        raise NotValid('non negative integer')
    if integer < 0:
        raise NotValid('non negative integer')
    return True