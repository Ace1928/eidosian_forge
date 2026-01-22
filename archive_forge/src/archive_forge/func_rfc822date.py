import base64
import binascii
import os
import random
import re
import socket
import time
import warnings
from email.utils import parseaddr
from io import BytesIO
from typing import Type
from zope.interface import implementer
from twisted import cred
from twisted.copyright import longversion
from twisted.internet import defer, error, protocol, reactor
from twisted.internet._idna import _idnaText
from twisted.internet.interfaces import ISSLTransport, ITLSTransport
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, util
from twisted.python.compat import iterbytes, nativeString, networkString
from twisted.python.runtime import platform
import codecs
def rfc822date(timeinfo=None, local=1):
    """
    Format an RFC-2822 compliant date string.

    @param timeinfo: (optional) A sequence as returned by C{time.localtime()}
        or C{time.gmtime()}. Default is now.
    @param local: (optional) Indicates if the supplied time is local or
        universal time, or if no time is given, whether now should be local or
        universal time. Default is local, as suggested (SHOULD) by rfc-2822.

    @returns: A L{bytes} representing the time and date in RFC-2822 format.
    """
    if not timeinfo:
        if local:
            timeinfo = time.localtime()
        else:
            timeinfo = time.gmtime()
    if local:
        if timeinfo[8]:
            tz = -time.altzone
        else:
            tz = -time.timezone
        tzhr, tzmin = divmod(abs(tz), 3600)
        if tz:
            tzhr *= int(abs(tz) // tz)
        tzmin, tzsec = divmod(tzmin, 60)
    else:
        tzhr, tzmin = (0, 0)
    return networkString('%s, %02d %s %04d %02d:%02d:%02d %+03d%02d' % (['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][timeinfo[6]], timeinfo[2], ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][timeinfo[1] - 1], timeinfo[0], timeinfo[3], timeinfo[4], timeinfo[5], tzhr, tzmin))