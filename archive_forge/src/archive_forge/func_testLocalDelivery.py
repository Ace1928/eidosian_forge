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
def testLocalDelivery(self):
    service = mail.mail.MailService()
    service.smtpPortal.registerChecker(cred.checkers.AllowAnonymousAccess())
    domain = mail.maildir.MaildirDirdbmDomain(service, 'domainDir')
    domain.addUser('user', 'password')
    service.addDomain('test.domain', domain)
    service.portals[''] = service.portals['test.domain']
    map(service.portals[''].registerChecker, domain.getCredentialsCheckers())
    service.setQueue(mail.relay.DomainQueuer(service))
    f = service.getSMTPFactory()
    self.smtpServer = reactor.listenTCP(0, f, interface='127.0.0.1')
    client = LineSendingProtocol(['HELO meson', 'MAIL FROM: <user@hostname>', 'RCPT TO: <user@test.domain>', 'DATA', 'This is the message', '.', 'QUIT'])
    done = Deferred()
    f = protocol.ClientFactory()
    f.protocol = lambda: client
    f.clientConnectionLost = lambda *args: done.callback(None)
    reactor.connectTCP('127.0.0.1', self.smtpServer.getHost().port, f)

    def finished(ign):
        mbox = domain.requestAvatar('user', None, pop3.IMailbox)[1]
        msg = mbox.getMessage(0).read()
        self.assertNotEqual(msg.find('This is the message'), -1)
        return self.smtpServer.stopListening()
    done.addCallback(finished)
    return done