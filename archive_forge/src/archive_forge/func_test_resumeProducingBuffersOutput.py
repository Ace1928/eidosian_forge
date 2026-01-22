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
def test_resumeProducingBuffersOutput(self):
    """
        L{MultipleRangeStaticProducer.start} writes about
        C{abstract.FileDescriptor.bufferSize} bytes of content from the
        resource to the request at once.

        To be specific about the 'about' above: it can write slightly more,
        for example in the case where the first boundary plus the first chunk
        is less than C{bufferSize} but first boundary plus the first chunk
        plus the second boundary is more, but this is unimportant as in
        practice the boundaries are fairly small.  On the other side, it is
        important for performance to bundle up several small chunks into one
        call to request.write.
        """
    request = DummyRequest([])
    content = b'0123456789' * 2
    producer = static.MultipleRangeStaticProducer(request, StringIO(content), [(b'a', 0, 2), (b'b', 5, 10), (b'c', 0, 0)])
    producer.bufferSize = 10
    producer.start()
    expected = [b'a' + content[0:2] + b'b' + content[5:11], content[11:15] + b'c']
    self.assertEqual(expected, request.written)