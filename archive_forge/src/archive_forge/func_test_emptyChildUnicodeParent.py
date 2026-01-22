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
def test_emptyChildUnicodeParent(self):
    """
        The C{u''} child of a L{File} which corresponds to a directory
        whose path is text is a L{DirectoryLister} that renders to a
        binary listing.

        @see: U{https://twistedmatrix.com/trac/ticket/9438}
        """
    textBase = FilePath(self.mktemp()).asTextMode()
    textBase.makedirs()
    textBase.child('text-file').open('w').close()
    textFile = static.File(textBase.path)
    request = DummyRequest([b''])
    child = resource.getChildForRequest(textFile, request)
    self.assertIsInstance(child, static.DirectoryLister)
    nativePath = compat.nativeString(textBase.path)
    self.assertEqual(child.path, nativePath)
    response = child.render(request)
    self.assertIsInstance(response, bytes)