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
def test__unwrap_text_file_read_mode(self):
    fd, file_path = tempfile.mkstemp()
    self.addCleanup(os.remove, file_path)
    fake_file = os.fdopen(fd, 'r')
    self.assertEqual(fake_file.buffer, subunit._unwrap_text(fake_file))