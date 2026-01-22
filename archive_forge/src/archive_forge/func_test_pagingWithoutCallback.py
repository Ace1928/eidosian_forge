import gc
import os
import sys
import time
import weakref
from collections import deque
from io import BytesIO as StringIO
from typing import Dict
from zope.interface import Interface, implementer
from twisted.cred import checkers, credentials, portal
from twisted.cred.error import UnauthorizedLogin, UnhandledCredentials
from twisted.internet import address, main, protocol, reactor
from twisted.internet.defer import Deferred, gatherResults, succeed
from twisted.internet.error import ConnectionRefusedError
from twisted.internet.testing import _FakeConnector
from twisted.protocols.policies import WrappingFactory
from twisted.python import failure, log
from twisted.python.compat import iterbytes
from twisted.spread import jelly, pb, publish, util
from twisted.trial import unittest
def test_pagingWithoutCallback(self):
    """
        Test L{util.StringPager} without a callback.
        """
    c, s, pump = connectedServerAndClient(test=self)
    s.setNameForLocal('foo', Pagerizer(None))
    x = c.remoteForName('foo')
    l = []
    util.getAllPages(x, 'getPages').addCallback(l.append)
    while not l:
        pump.pump()
    self.assertEqual(b''.join(l[0]), bigString, 'Pages received not equal to pages sent!')