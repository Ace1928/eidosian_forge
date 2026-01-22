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
def testFileLoader(self):
    domains = {'': object()}
    result = mail.alias.loadAliasFile(domains, fp=io.BytesIO(textwrap.dedent("                    # Here's a comment\n                       # woop another one\n                    testuser:                   address1,address2, address3,\n                        continuation@address, |/bin/process/this\n\n                    usertwo:thisaddress,thataddress, lastaddress\n                    lastuser:       :/includable, /filename, |/program, address\n                    ").encode()))
    self.assertEqual(len(result), 3)
    group = result['testuser']
    s = str(group)
    for a in ('address1', 'address2', 'address3', 'continuation@address', '/bin/process/this'):
        self.assertNotEqual(s.find(a), -1)
    self.assertEqual(len(group), 5)
    group = result['usertwo']
    s = str(group)
    for a in ('thisaddress', 'thataddress', 'lastaddress'):
        self.assertNotEqual(s.find(a), -1)
    self.assertEqual(len(group), 3)
    group = result['lastuser']
    s = str(group)
    self.assertEqual(s.find('/includable'), -1)
    for a in ('/filename', 'program', 'address'):
        self.assertNotEqual(s.find(a), -1, '%s not found' % a)
    self.assertEqual(len(group), 3)