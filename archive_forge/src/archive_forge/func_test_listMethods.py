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
def test_listMethods(self):

    def cbMethods(meths):
        meths.sort()
        self.assertEqual(meths, ['add', 'complex', 'defer', 'deferFail', 'deferFault', 'dict', 'echo', 'fail', 'fault', 'pair', 'snowman', 'system.listMethods', 'system.methodHelp', 'system.methodSignature', 'withRequest'])
    d = self.proxy().callRemote('system.listMethods')
    d.addCallback(cbMethods)
    return d