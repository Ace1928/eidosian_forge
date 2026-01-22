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
def do_RENAME(self, tag, oldname, newname):
    oldname, newname = (_parseMbox(n) for n in (oldname, newname))
    if oldname.lower() == 'inbox' or newname.lower() == 'inbox':
        self.sendNegativeResponse(tag, b'You cannot rename the inbox, or rename another mailbox to inbox.')
        return
    try:
        self.account.rename(oldname, newname)
    except TypeError:
        self.sendBadResponse(tag, b'Invalid command syntax')
    except MailboxException as m:
        self.sendNegativeResponse(tag, networkString(str(m)))
    except BaseException:
        self.sendBadResponse(tag, b'Server error encountered while renaming mailbox')
        log.err()
    else:
        self.sendPositiveResponse(tag, b'Mailbox renamed')