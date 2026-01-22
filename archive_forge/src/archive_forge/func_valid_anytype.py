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
def valid_anytype(val):
    """Goes through all known type validators

    :param val: The value to validate
    :return: True is value is valid otherwise an exception is raised
    """
    for validator in VALIDATOR.values():
        if validator == valid_anytype:
            continue
        try:
            if validator(val):
                return True
        except NotValid:
            pass
    if isinstance(val, type):
        return True
    raise NotValid('AnyType')