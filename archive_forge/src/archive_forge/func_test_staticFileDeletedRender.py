import errno
import inspect
import mimetypes
import os
import re
import sys
import warnings
from io import BytesIO as StringIO
from unittest import skipIf
from zope.interface.verify import verifyObject
from twisted.internet import abstract, interfaces
from twisted.python import compat, log
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.web import http, resource, script, static
from twisted.web._responses import FOUND
from twisted.web.server import UnsupportedMethod
from twisted.web.test._util import _render
from twisted.web.test.requesthelper import DummyRequest
def test_staticFileDeletedRender(self):
    """
        A L{static.File} created for a file which does not exist should render
        its C{childNotFound} page.
        """
    staticFile = static.File(self.mktemp())
    request = DummyRequest([b'foo.bar'])
    request2 = DummyRequest([b'foo.bar'])
    d = self._render(staticFile, request)
    d2 = self._render(staticFile.childNotFound, request2)

    def cbRendered2(ignored):

        def cbRendered(ignored):
            self.assertEqual(b''.join(request.written), b''.join(request2.written))
        d.addCallback(cbRendered)
        return d
    d2.addCallback(cbRendered2)
    return d2