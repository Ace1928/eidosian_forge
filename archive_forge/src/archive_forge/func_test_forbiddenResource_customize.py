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
def test_forbiddenResource_customize(self):
    """
        The resource rendered for forbidden requests is stored as a class
        member so that users can customize it.
        """
    base = FilePath(self.mktemp())
    base.setContent(b'')
    markerResponse = b'custom-forbidden-response'

    def failingOpenForReading():
        raise OSError(errno.EACCES, '')

    class CustomForbiddenResource(resource.Resource):

        def render(self, request):
            return markerResponse

    class CustomStaticFile(static.File):
        forbidden = CustomForbiddenResource()
    fileResource = CustomStaticFile(base.path)
    fileResource.openForReading = failingOpenForReading
    request = DummyRequest([b''])
    result = fileResource.render(request)
    self.assertEqual(markerResponse, result)