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
def test_add_unexpected_success_details(self):
    """Test addUnexpectedSuccess on a TestProtocolClient with details."""
    self.protocol.addUnexpectedSuccess(self.test, details=self.sample_details)
    self.assertEqual(self.io.getvalue(), _b('uxsuccess: %s [ multipart\nContent-Type: text/plain\nsomething\nF\r\nserialised\nform0\r\n]\n' % self.test.id()))