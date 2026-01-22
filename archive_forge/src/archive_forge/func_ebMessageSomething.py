import base64
import binascii
import warnings
from hashlib import md5
from typing import Optional
from zope.interface import implementer
from twisted import cred
from twisted.internet import defer, interfaces, task
from twisted.mail import smtp
from twisted.mail._except import POP3ClientError, POP3Error, _POP3MessageDeleted
from twisted.mail.interfaces import (
from twisted.protocols import basic, policies
from twisted.python import log
from twisted.mail._except import (
from twisted.mail._pop3client import POP3Client as AdvancedPOP3Client
def ebMessageSomething(err):
    errcls = err.check(_POP3MessageDeleted, ValueError, IndexError)
    if errcls is _POP3MessageDeleted:
        self.failResponse('message deleted')
    elif errcls in (ValueError, IndexError):
        if errcls is IndexError:
            warnings.warn('twisted.mail.pop3.IMailbox.listMessages may not raise IndexError for out-of-bounds message numbers: raise ValueError instead.', PendingDeprecationWarning)
        self.failResponse('Bad message number argument')
    else:
        log.msg('Unexpected _getMessageFile failure:')
        log.err(err)
    return None