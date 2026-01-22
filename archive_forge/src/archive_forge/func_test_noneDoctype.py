import sys
from io import StringIO
from typing import List, Optional
from zope.interface import implementer
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, succeed
from twisted.internet.testing import EventLoggingObserver
from twisted.logger import globalLogPublisher
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.trial.unittest import TestCase
from twisted.trial.util import suppress as SUPPRESS
from twisted.web._element import UnexposedMethodError
from twisted.web.error import FlattenerError, MissingRenderMethod, MissingTemplateLoader
from twisted.web.iweb import IRequest, ITemplateLoader
from twisted.web.server import NOT_DONE_YET
from twisted.web.template import (
from twisted.web.test._util import FlattenTestCase
from twisted.web.test.test_web import DummyRequest
def test_noneDoctype(self) -> Deferred[None]:
    """
        L{renderElement} will not write out a doctype if the doctype keyword
        argument is L{None}.
        """
    element = TestElement()
    d = self.request.notifyFinish()

    def check(_: object) -> None:
        self.assertEqual(b''.join(self.request.written), b'<p>Hello, world.</p>')
    d.addCallback(check)
    renderElement(self.request, element, doctype=None)
    return d