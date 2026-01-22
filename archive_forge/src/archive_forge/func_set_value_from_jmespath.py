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
def set_value_from_jmespath(source, expression, value, is_first=True):
    if is_first:
        validate_jmespath_for_set(expression)
    bits = expression.split('.', 1)
    current_key, remainder = (bits[0], bits[1] if len(bits) > 1 else '')
    if not current_key:
        raise InvalidExpressionError(expression=expression)
    if remainder:
        if current_key not in source:
            source[current_key] = {}
        return set_value_from_jmespath(source[current_key], remainder, value, is_first=False)
    source[current_key] = value