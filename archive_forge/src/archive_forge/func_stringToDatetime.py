from __future__ import annotations
import base64
import binascii
import calendar
import math
import os
import re
import tempfile
import time
import warnings
from email import message_from_bytes
from email.message import EmailMessage
from io import BytesIO
from typing import AnyStr, Callable, Dict, List, Optional, Tuple
from urllib.parse import (
from zope.interface import Attribute, Interface, implementer, provider
from incremental import Version
from twisted.internet import address, interfaces, protocol
from twisted.internet._producer_helpers import _PullToPush
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IProtocol
from twisted.logger import Logger
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.python.compat import nativeString, networkString
from twisted.python.components import proxyForInterface
from twisted.python.deprecate import deprecated
from twisted.python.failure import Failure
from twisted.web._responses import (
from twisted.web.http_headers import Headers, _sanitizeLinearWhitespace
from twisted.web.iweb import IAccessLogFormatter, INonQueuedRequestFactory, IRequest
def stringToDatetime(dateString):
    """
    Convert an HTTP date string (one of three formats) to seconds since epoch.

    @type dateString: C{bytes}
    """
    parts = nativeString(dateString).split()
    if not parts[0][0:3].lower() in weekdayname_lower:
        try:
            return stringToDatetime(b'Sun, ' + dateString)
        except ValueError:
            pass
    partlen = len(parts)
    if (partlen == 5 or partlen == 6) and parts[1].isdigit():
        day = parts[1]
        month = parts[2]
        year = parts[3]
        time = parts[4]
    elif (partlen == 3 or partlen == 4) and parts[1].find('-') != -1:
        day, month, year = parts[1].split('-')
        time = parts[2]
        year = int(year)
        if year < 69:
            year = year + 2000
        elif year < 100:
            year = year + 1900
    elif len(parts) == 5:
        day = parts[2]
        month = parts[1]
        year = parts[4]
        time = parts[3]
    else:
        raise ValueError('Unknown datetime format %r' % dateString)
    day = int(day)
    month = int(monthname_lower.index(month.lower()))
    year = int(year)
    hour, min, sec = map(int, time.split(':'))
    return int(timegm(year, month, day, hour, min, sec))