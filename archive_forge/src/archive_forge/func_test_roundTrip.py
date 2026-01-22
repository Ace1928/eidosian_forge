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
def test_roundTrip(self) -> None:
    """
        Given a series of parsable XML strings, verify that
        L{twisted.web._flatten.flatten} will flatten the L{Element} back to the
        input when sent on a round trip.
        """
    fragments = [b'<p>Hello, world.</p>', b'<p><!-- hello, world --></p>', b'<p><![CDATA[Hello, world.]]></p>', b'<test1 xmlns:test2="urn:test2"><test2:test3></test2:test3></test1>', b'<test1 xmlns="urn:test2"><test3></test3></test1>', b'<p>\xe2\x98\x83</p>']
    for xml in fragments:
        self.assertFlattensImmediately(Element(loader=XMLString(xml)), xml)