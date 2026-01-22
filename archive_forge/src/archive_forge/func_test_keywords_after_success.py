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
def test_keywords_after_success(self):
    self.protocol.lineReceived(_b('test old mcdonald\n'))
    self.protocol.lineReceived(_b('success old mcdonald\n'))
    self.keywords_before_test()
    self.assertEqual([('startTest', self.test), ('addSuccess', self.test), ('stopTest', self.test)], self.client._events)