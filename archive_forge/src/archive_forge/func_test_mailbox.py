import email.message
import email.parser
import errno
import glob
import io
import os
import pickle
import shutil
import signal
import sys
import tempfile
import textwrap
import time
from hashlib import md5
from unittest import skipIf
from zope.interface import Interface, implementer
from zope.interface.verify import verifyClass
import twisted.cred.checkers
import twisted.cred.credentials
import twisted.cred.portal
import twisted.mail.alias
import twisted.mail.mail
import twisted.mail.maildir
import twisted.mail.protocols
import twisted.mail.relay
import twisted.mail.relaymanager
from twisted import cred, mail
from twisted.internet import address, defer, interfaces, protocol, reactor, task
from twisted.internet.defer import Deferred
from twisted.internet.error import (
from twisted.internet.testing import (
from twisted.mail import pop3, smtp
from twisted.mail.relaymanager import _AttemptManager
from twisted.names import dns
from twisted.names.dns import Record_CNAME, Record_MX, RRHeader
from twisted.names.error import DNSNameError
from twisted.python import failure, log
from twisted.python.filepath import FilePath
from twisted.python.runtime import platformType
from twisted.trial.unittest import TestCase
from twisted.names import client, common, server
@skipIf(sys.version_info >= (3,), 'not ported to Python 3')
def test_mailbox(self):
    """
        Exercise the methods of L{IMailbox} as implemented by
        L{MaildirMailbox}.
        """
    j = os.path.join
    n = mail.maildir._generateMaildirName
    msgs = [j(b, n()) for b in ('cur', 'new') for x in range(5)]
    i = 1
    for f in msgs:
        with open(j(self.d, f), 'w') as fObj:
            fObj.write('x' * i)
        i = i + 1
    mb = mail.maildir.MaildirMailbox(self.d)
    self.assertEqual(mb.listMessages(), list(range(1, 11)))
    self.assertEqual(mb.listMessages(1), 2)
    self.assertEqual(mb.listMessages(5), 6)
    self.assertEqual(mb.getMessage(6).read(), 'x' * 7)
    self.assertEqual(mb.getMessage(1).read(), 'x' * 2)
    d = {}
    for i in range(10):
        u = mb.getUidl(i)
        self.assertFalse(u in d)
        d[u] = None
    p, f = os.path.split(msgs[5])
    mb.deleteMessage(5)
    self.assertEqual(mb.listMessages(5), 0)
    self.assertTrue(os.path.exists(j(self.d, '.Trash', 'cur', f)))
    self.assertFalse(os.path.exists(j(self.d, msgs[5])))
    mb.undeleteMessages()
    self.assertEqual(mb.listMessages(5), 6)
    self.assertFalse(os.path.exists(j(self.d, '.Trash', 'cur', f)))
    self.assertTrue(os.path.exists(j(self.d, msgs[5])))