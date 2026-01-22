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
def do_STARTTLS(self, tag):
    if self.startedTLS:
        self.sendNegativeResponse(tag, b'TLS already negotiated')
    elif self.ctx and self.canStartTLS:
        self.sendPositiveResponse(tag, b'Begin TLS negotiation now')
        self.transport.startTLS(self.ctx)
        self.startedTLS = True
        self.challengers = self.challengers.copy()
        if b'LOGIN' not in self.challengers:
            self.challengers[b'LOGIN'] = LOGINCredentials
        if b'PLAIN' not in self.challengers:
            self.challengers[b'PLAIN'] = PLAINCredentials
    else:
        self.sendNegativeResponse(tag, b'TLS not available')