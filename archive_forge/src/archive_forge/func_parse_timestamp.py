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
def parse_timestamp(value):
    """Parse a timestamp into a datetime object.

    Supported formats:

        * iso8601
        * rfc822
        * epoch (value is an integer)

    This will return a ``datetime.datetime`` object.

    """
    tzinfo_options = get_tzinfo_options()
    for tzinfo in tzinfo_options:
        try:
            return _parse_timestamp_with_tzinfo(value, tzinfo)
        except (OSError, OverflowError) as e:
            logger.debug('Unable to parse timestamp with "%s" timezone info.', tzinfo.__name__, exc_info=e)
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        pass
    else:
        try:
            for tzinfo in tzinfo_options:
                return _epoch_seconds_to_datetime(numeric_value, tzinfo=tzinfo)
        except (OSError, OverflowError) as e:
            logger.debug('Unable to parse timestamp using fallback method with "%s" timezone info.', tzinfo.__name__, exc_info=e)
    raise RuntimeError('Unable to calculate correct timezone offset for "%s"' % value)