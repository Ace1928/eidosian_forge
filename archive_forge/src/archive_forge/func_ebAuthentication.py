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
def ebAuthentication(err):
    """
                Translate cred exceptions into SMTP exceptions so that the
                protocol code which invokes C{validateFrom} can properly report
                the failure.
                """
    if err.check(cred.error.UnauthorizedLogin):
        exc = SMTPBadSender(origin)
    elif err.check(cred.error.UnhandledCredentials):
        exc = SMTPBadSender(origin, resp='Unauthenticated senders not allowed')
    else:
        return err
    return defer.fail(exc)