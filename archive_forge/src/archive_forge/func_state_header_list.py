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
def state_header_list(self, s):
    if not s.startswith(b'('):
        raise Exception('Header list must begin with (')
    end = s.find(b')')
    if end == -1:
        raise Exception('Header list must end with )')
    headers = s[1:end].split()
    self.pending_body.header.fields = [h.upper() for h in headers]
    return end + 1