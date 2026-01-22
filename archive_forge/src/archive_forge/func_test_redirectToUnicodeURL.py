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
def test_redirectToUnicodeURL(self):
    """
        L{redirectTo} will raise TypeError if unicode object is passed in URL
        """
    request = Request(DummyChannel(), True)
    request.method = b'GET'
    targetURL = 'http://target.example.com/4321'
    self.assertRaises(TypeError, redirectTo, targetURL, request)