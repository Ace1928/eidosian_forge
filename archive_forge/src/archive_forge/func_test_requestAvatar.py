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
def test_requestAvatar(self):
    """
        L{MaildirDirdbmDomain.requestAvatar} raises L{NotImplementedError}
        unless it is supplied with an L{pop3.IMailbox} interface.
        When called with an L{pop3.IMailbox}, it returns a 3-tuple
        containing L{pop3.IMailbox}, an implementation of that interface
        and a NOOP callable.
        """

    class ISomething(Interface):
        pass
    self.D.addUser('user', 'password')
    self.assertRaises(NotImplementedError, self.D.requestAvatar, 'user', None, ISomething)
    t = self.D.requestAvatar('user', None, pop3.IMailbox)
    self.assertEqual(len(t), 3)
    self.assertTrue(t[0] is pop3.IMailbox)
    self.assertTrue(pop3.IMailbox.providedBy(t[1]))
    t[2]()