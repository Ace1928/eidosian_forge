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
def state_initial(self, s):
    if s == b'':
        return 0
    l = s.lower()
    if l.startswith(b'all'):
        self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope()))
        return 3
    if l.startswith(b'full'):
        self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size(), self.Envelope(), self.Body()))
        return 4
    if l.startswith(b'fast'):
        self.result.extend((self.Flags(), self.InternalDate(), self.RFC822Size()))
        return 4
    if l.startswith(b'('):
        self.state.extend(('close_paren', 'maybe_fetch_att', 'fetch_att'))
        return 1
    self.state.append('fetch_att')
    return 0