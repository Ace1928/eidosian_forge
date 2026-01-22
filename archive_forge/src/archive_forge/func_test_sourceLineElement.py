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
def test_sourceLineElement(self):
    """
        L{_SourceLineElement} renders a source line and line number.
        """
    element = _SourceLineElement(TagLoader(tags.div(tags.span(render='lineNumber'), tags.span(render='sourceLine'))), 50, "    print 'hello'")
    d = flattenString(None, element)
    expected = "<div><span>50</span><span> \xa0 \xa0print 'hello'</span></div>"
    d.addCallback(self.assertEqual, expected.encode('utf-8'))
    return d