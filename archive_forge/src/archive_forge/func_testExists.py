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
def testExists(self):
    service = mail.mail.MailService()
    domain = mail.relay.DomainQueuer(service)
    doRelay = [address.UNIXAddress('/var/run/mail-relay'), address.IPv4Address('TCP', '127.0.0.1', 12345)]
    dontRelay = [address.IPv4Address('TCP', '192.168.2.1', 62), address.IPv4Address('TCP', '1.2.3.4', 1943)]
    for peer in doRelay:
        user = empty()
        user.orig = 'user@host'
        user.dest = 'tsoh@resu'
        user.protocol = empty()
        user.protocol.transport = empty()
        user.protocol.transport.getPeer = lambda: peer
        self.assertTrue(callable(domain.exists(user)))
    for peer in dontRelay:
        user = empty()
        user.orig = 'some@place'
        user.protocol = empty()
        user.protocol.transport = empty()
        user.protocol.transport.getPeer = lambda: peer
        user.dest = 'who@cares'
        self.assertRaises(smtp.SMTPBadRcpt, domain.exists, user)