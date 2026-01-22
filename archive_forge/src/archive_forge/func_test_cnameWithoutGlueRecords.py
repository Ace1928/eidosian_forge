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
def test_cnameWithoutGlueRecords(self):
    """
        If an MX lookup returns a single CNAME record as a result, MXCalculator
        will perform an MX lookup for the canonical name indicated and return
        the MX record which results.
        """
    alias = 'alias.example.com'
    canonical = 'canonical.example.com'
    exchange = 'mail.example.com'

    class DummyResolver:
        """
            Fake resolver which will return a CNAME for an MX lookup of a name
            which is an alias and an MX for an MX lookup of the canonical name.
            """

        def lookupMailExchange(self, domain):
            if domain == alias:
                return defer.succeed(([RRHeader(name=domain, type=Record_CNAME.TYPE, payload=Record_CNAME(canonical))], [], []))
            elif domain == canonical:
                return defer.succeed(([RRHeader(name=domain, type=Record_MX.TYPE, payload=Record_MX(0, exchange))], [], []))
            else:
                return defer.fail(DNSNameError(domain))
    self.mx.resolver = DummyResolver()
    d = self.mx.getMX(alias)
    d.addCallback(self.assertEqual, Record_MX(name=exchange))
    return d