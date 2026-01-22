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
def ext_STARTTLS(self, rest):
    if self.startedTLS:
        self.sendCode(503, b'TLS already negotiated')
    elif self.ctx and self.canStartTLS:
        self.sendCode(220, b'Begin TLS negotiation now')
        self.transport.startTLS(self.ctx)
        self.startedTLS = True
    else:
        self.sendCode(454, b'TLS not available')