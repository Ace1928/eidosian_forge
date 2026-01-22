import binascii
import codecs
import copy
import email.utils
import functools
import re
import string
import tempfile
import time
import uuid
from base64 import decodebytes, encodebytes
from io import BytesIO
from itertools import chain
from typing import Any, List, cast
from zope.interface import implementer
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import defer, error, interfaces
from twisted.internet.defer import maybeDeferred
from twisted.mail._cred import (
from twisted.mail._except import (
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log, text
from twisted.python.compat import (
def spew_internaldate(self, id, msg, _w=None, _f=None):
    if _w is None:
        _w = self.transport.write
    idate = msg.getInternalDate()
    ttup = email.utils.parsedate_tz(nativeString(idate))
    if ttup is None:
        log.msg('%d:%r: unpareseable internaldate: %r' % (id, msg, idate))
        raise IMAP4Exception('Internal failure generating INTERNALDATE')
    strdate = time.strftime('%d-%%s-%Y %H:%M:%S ', ttup[:9])
    odate = networkString(strdate % (_MONTH_NAMES[ttup[1]],))
    if ttup[9] is None:
        odate = odate + b'+0000'
    else:
        if ttup[9] >= 0:
            sign = b'+'
        else:
            sign = b'-'
        odate = odate + sign + b'%04d' % (abs(ttup[9]) // 3600 * 100 + abs(ttup[9]) % 3600 // 60,)
    _w(b'INTERNALDATE ' + _quote(odate))