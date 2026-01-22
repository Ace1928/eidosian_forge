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
def test_escapeHeader(self):
    """
        L{static.DirectoryLister} escape "&", "<" and ">" after unquoting the
        request uri.
        """
    path = FilePath(self.mktemp())
    path.makedirs()
    lister = static.DirectoryLister(path.path)
    data = lister.render(self._request(b'foo%26bar'))
    self.assertIn(b'<h1>Directory listing for foo&amp;bar</h1>', data)
    self.assertIn(b'<title>Directory listing for foo&amp;bar</title>', data)