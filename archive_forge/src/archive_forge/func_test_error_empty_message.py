import datetime
import io
import os
import tempfile
import unittest
from io import BytesIO
from testtools import PlaceHolder, TestCase, TestResult, skipIf
from testtools.compat import _b, _u
from testtools.content import Content, TracebackContent, text_content
from testtools.content_type import ContentType
from testtools.matchers import Contains, Equals, MatchesAny
import iso8601
import subunit
from subunit.tests import (_remote_exception_repr,
def test_error_empty_message(self):
    self.protocol.lineReceived(_b('error mcdonalds farm [\n'))
    self.protocol.lineReceived(_b(']\n'))
    details = {}
    details['traceback'] = Content(ContentType('text', 'x-traceback', {'charset': 'utf8'}), lambda: [_b('')])
    self.assertEqual([('startTest', self.test), ('addError', self.test, details), ('stopTest', self.test)], self.client._events)