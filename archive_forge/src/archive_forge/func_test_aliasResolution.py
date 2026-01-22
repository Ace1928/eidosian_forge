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
def test_aliasResolution(self):
    """
        Check that the C{resolve} method of alias processors produce the correct
        set of objects:
            - direct alias with L{mail.alias.AddressAlias} if a simple input is passed
            - aliases in a file with L{mail.alias.FileWrapper} if an input in the format
              '/file' is given
            - aliases resulting of a process call wrapped by L{mail.alias.MessageWrapper}
              if the format is '|process'
        """
    aliases = {}
    domain = {'': TestDomain(aliases, ['user1', 'user2', 'user3'])}
    A1 = MockAliasGroup(['user1', '|echo', '/file'], domain, 'alias1')
    A2 = MockAliasGroup(['user2', 'user3'], domain, 'alias2')
    A3 = mail.alias.AddressAlias('alias1', domain, 'alias3')
    aliases.update({'alias1': A1, 'alias2': A2, 'alias3': A3})
    res1 = A1.resolve(aliases)
    r1 = map(str, res1.objs)
    r1.sort()
    expected = map(str, [mail.alias.AddressAlias('user1', None, None), mail.alias.MessageWrapper(DummyProcess(), 'echo'), mail.alias.FileWrapper('/file')])
    expected.sort()
    self.assertEqual(r1, expected)
    res2 = A2.resolve(aliases)
    r2 = map(str, res2.objs)
    r2.sort()
    expected = map(str, [mail.alias.AddressAlias('user2', None, None), mail.alias.AddressAlias('user3', None, None)])
    expected.sort()
    self.assertEqual(r2, expected)
    res3 = A3.resolve(aliases)
    r3 = map(str, res3.objs)
    r3.sort()
    expected = map(str, [mail.alias.AddressAlias('user1', None, None), mail.alias.MessageWrapper(DummyProcess(), 'echo'), mail.alias.FileWrapper('/file')])
    expected.sort()
    self.assertEqual(r3, expected)