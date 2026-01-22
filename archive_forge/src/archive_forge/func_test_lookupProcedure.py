import datetime
from io import BytesIO, StringIO
from unittest import skipIf
from twisted.internet import defer, reactor
from twisted.internet.error import ConnectionDone
from twisted.internet.testing import EventLoggingObserver, MemoryReactor
from twisted.logger import (
from twisted.python import failure
from twisted.python.compat import nativeString, networkString
from twisted.python.reflect import namedModule
from twisted.trial import unittest
from twisted.web import client, http, server, static, xmlrpc
from twisted.web.test.test_web import DummyRequest
from twisted.web.xmlrpc import (
def test_lookupProcedure(self):
    """
        A subclass of L{XMLRPC} can override C{lookupProcedure} to find
        procedures that are not defined using a C{xmlrpc_}-prefixed method name.
        """
    self.createServer(TestLookupProcedure())
    what = 'hello'
    d = self.proxy.callRemote('echo', what)
    d.addCallback(self.assertEqual, what)
    return d