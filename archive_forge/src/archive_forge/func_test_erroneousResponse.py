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
def test_erroneousResponse(self):
    """
        Test that calling the xmlrpc client on a static http server raises
        an exception.
        """
    proxy = xmlrpc.Proxy(networkString('http://127.0.0.1:%d/' % (self.port.getHost().port,)))
    return self.assertFailure(proxy.callRemote('someMethod'), ValueError)