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
def parseIdList(s, lastMessageId=None):
    """
    Parse a message set search key into a C{MessageSet}.

    @type s: L{bytes}
    @param s: A string description of an id list, for example "1:3, 4:*"

    @type lastMessageId: L{int}
    @param lastMessageId: The last message sequence id or UID, depending on
        whether we are parsing the list in UID or sequence id context. The
        caller should pass in the correct value.

    @rtype: C{MessageSet}
    @return: A C{MessageSet} that contains the ids defined in the list
    """
    res = MessageSet()
    parts = s.split(b',')
    for p in parts:
        if b':' in p:
            low, high = p.split(b':', 1)
            try:
                if low == b'*':
                    low = None
                else:
                    low = int(low)
                if high == b'*':
                    high = None
                else:
                    high = int(high)
                if low is high is None:
                    raise IllegalIdentifierError(p)
                if low is not None and low <= 0 or (high is not None and high <= 0):
                    raise IllegalIdentifierError(p)
                high = high or lastMessageId
                low = low or lastMessageId
                res.add(low, high)
            except ValueError:
                raise IllegalIdentifierError(p)
        else:
            try:
                if p == b'*':
                    p = None
                else:
                    p = int(p)
                if p is not None and p <= 0:
                    raise IllegalIdentifierError(p)
            except ValueError:
                raise IllegalIdentifierError(p)
            else:
                res.extend(p or lastMessageId)
    return res