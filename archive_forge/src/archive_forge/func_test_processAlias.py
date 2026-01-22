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
def test_processAlias(self):
    """
        Standard call to C{mail.alias.ProcessAlias}: check that the specified
        script is called, and that the input is correctly transferred to it.
        """
    sh = FilePath(self.mktemp())
    sh.setContent('#!/bin/sh\nrm -f process.alias.out\nwhile read i; do\n    echo $i >> process.alias.out\ndone')
    os.chmod(sh.path, 448)
    a = mail.alias.ProcessAlias(sh.path, None, None)
    m = a.createMessageReceiver()
    for l in self.lines:
        m.lineReceived(l)

    def _cbProcessAlias(ignored):
        with open('process.alias.out') as f:
            lines = f.readlines()
        self.assertEqual([L[:-1] for L in lines], self.lines)
    return m.eomReceived().addCallback(_cbProcessAlias)