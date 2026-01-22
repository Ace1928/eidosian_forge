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
def test_explicitAuthInfoOverride(self):
    p = xmlrpc.Proxy(networkString('http://wrong:info@127.0.0.1:%d/' % (self.port,)), self.user, self.password)
    d = p.callRemote('authinfo')
    d.addCallback(self.assertEqual, [self.user, self.password])
    return d