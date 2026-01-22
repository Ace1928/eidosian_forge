import gc
from twisted.internet import defer
from twisted.python.compat import networkString
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
from twisted.web import resource, util
from twisted.web.error import FlattenerError
from twisted.web.http import FOUND
from twisted.web.server import Request
from twisted.web.template import TagLoader, flattenString, tags
from twisted.web.test.requesthelper import DummyChannel, DummyRequest
from twisted.web.util import (
def test_frameElementFunction(self):
    """
        The I{function} renderer of L{_FrameElement} renders the line number
        associated with the frame object used to initialize the
        L{_FrameElement}.
        """
    element = _FrameElement(TagLoader(tags.span(render='function')), self.frame)
    d = flattenString(None, element)
    d.addCallback(self.assertEqual, b'<span>lineNumberProbeAlsoBroken</span>')
    return d