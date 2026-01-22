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
def test_invalidRanges(self):
    """
        L{File._parseRangeHeader} raises L{ValueError} when passed
        syntactically invalid byte ranges.
        """
    f = self.resource._parseRangeHeader
    self.assertRaises(ValueError, f, b'bytes')
    self.assertRaises(ValueError, f, b'unknown=1-2')
    self.assertRaises(ValueError, f, b'bytes=3')
    self.assertRaises(ValueError, f, b'bytes=-')
    self.assertRaises(ValueError, f, b'bytes=foo-')
    self.assertRaises(ValueError, f, b'bytes=-foo')
    self.assertRaises(ValueError, f, b'bytes=5-4')