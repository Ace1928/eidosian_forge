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
def test_ignoredExtensionsIgnored(self):
    """
        A request for the I{base} child of a L{File} succeeds with a resource
        for the I{base<extension>} file in the path the L{File} was created
        with if such a file exists and the L{File} has been configured to
        ignore the I{<extension>} extension.
        """
    base = FilePath(self.mktemp())
    base.makedirs()
    base.child('foo.bar').setContent(b'baz')
    base.child('foo.quux').setContent(b'foobar')
    file = static.File(base.path, ignoredExts=('.bar',))
    request = DummyRequest([b'foo'])
    child = resource.getChildForRequest(file, request)
    d = self._render(child, request)

    def cbRendered(ignored):
        self.assertEqual(b''.join(request.written), b'baz')
    d.addCallback(cbRendered)
    return d