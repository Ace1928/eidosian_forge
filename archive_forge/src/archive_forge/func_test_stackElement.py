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
def test_stackElement(self):
    """
        The I{frames} renderer of L{_StackElement} renders each stack frame in
        the list of frames used to initialize the L{_StackElement}.
        """
    element = _StackElement(None, self.failure.frames[:2])
    renderer = element.lookupRenderMethod('frames')
    tag = tags.div()
    result = renderer(None, tag)
    self.assertIsInstance(result, list)
    self.assertIsInstance(result[0], _FrameElement)
    self.assertIdentical(result[0].frame, self.failure.frames[0])
    self.assertIsInstance(result[1], _FrameElement)
    self.assertIdentical(result[1].frame, self.failure.frames[1])
    self.assertNotEqual(result[0].loader.load(), result[1].loader.load())
    self.assertEqual(2, len(result))