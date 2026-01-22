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
def do_STORE(self, tag, messages, mode, flags, uid=0):
    mode = mode.upper()
    silent = mode.endswith(b'SILENT')
    if mode.startswith(b'+'):
        mode = 1
    elif mode.startswith(b'-'):
        mode = -1
    else:
        mode = 0
    flags = [nativeString(flag) for flag in flags]
    maybeDeferred(self.mbox.store, messages, flags, mode, uid=uid).addCallbacks(self.__cbStore, self.__ebStore, (tag, self.mbox, uid, silent), None, (tag,), None)