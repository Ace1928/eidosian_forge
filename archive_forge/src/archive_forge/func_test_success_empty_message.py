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
def test_success_empty_message(self):
    self.protocol.lineReceived(_b('success mcdonalds farm [\n'))
    self.protocol.lineReceived(_b(']\n'))
    details = {}
    details['message'] = Content(ContentType('text', 'plain'), lambda: [_b('')])
    self.assertSuccess(details)