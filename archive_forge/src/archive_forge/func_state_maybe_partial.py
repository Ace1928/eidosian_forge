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
def state_maybe_partial(self, s):
    if not s.startswith(b'<'):
        return 0
    end = s.find(b'>')
    if end == -1:
        raise Exception('Found < but not >')
    partial = s[1:end]
    parts = partial.split(b'.', 1)
    if len(parts) != 2:
        raise Exception('Partial specification did not include two .-delimited integers')
    begin, length = map(int, parts)
    self.pending_body.partialBegin = begin
    self.pending_body.partialLength = length
    return end + 1