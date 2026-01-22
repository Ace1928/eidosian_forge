import base64
import binascii
import datetime
import email.message
import functools
import hashlib
import io
import logging
import os
import random
import re
import socket
import time
import warnings
import weakref
from datetime import datetime as _DatetimeClass
from ipaddress import ip_address
from pathlib import Path
from urllib.request import getproxies, proxy_bypass
import dateutil.parser
from dateutil.tz import tzutc
from urllib3.exceptions import LocationParseError
import botocore
import botocore.awsrequest
import botocore.httpsession
from botocore.compat import HEX_PAT  # noqa: F401
from botocore.compat import IPV4_PAT  # noqa: F401
from botocore.compat import IPV6_ADDRZ_PAT  # noqa: F401
from botocore.compat import IPV6_PAT  # noqa: F401
from botocore.compat import LS32_PAT  # noqa: F401
from botocore.compat import UNRESERVED_PAT  # noqa: F401
from botocore.compat import ZONE_ID_PAT  # noqa: F401
from botocore.compat import (
from botocore.exceptions import (
def percent_encode(input_str, safe=SAFE_CHARS):
    """Urlencodes a string.

    Whereas percent_encode_sequence handles taking a dict/sequence and
    producing a percent encoded string, this function deals only with
    taking a string (not a dict/sequence) and percent encoding it.

    If given the binary type, will simply URL encode it. If given the
    text type, will produce the binary type by UTF-8 encoding the
    text. If given something else, will convert it to the text type
    first.
    """
    if not isinstance(input_str, (bytes, str)):
        input_str = str(input_str)
    if not isinstance(input_str, bytes):
        input_str = input_str.encode('utf-8')
    return quote(input_str, safe=safe)