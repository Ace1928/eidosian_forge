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
def dequote(self, addr):
    """
        Remove RFC-2821 quotes from address.
        """
    res = []
    if not isinstance(addr, bytes):
        addr = str(addr).encode('ascii')
    atl = filter(None, self.tstring.split(addr))
    for t in atl:
        if t[0] == b'"' and t[-1] == b'"':
            res.append(t[1:-1])
        elif '\\' in t:
            res.append(self.dequotebs.sub(b'\\1', t))
        else:
            res.append(t)
    return b''.join(res)