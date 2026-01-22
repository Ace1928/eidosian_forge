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
def test_tags_do_not_get_set_on_test(self):
    self.protocol.lineReceived(_b('test mcdonalds farm\n'))
    test = self.client._events[0][-1]
    self.assertEqual(None, getattr(test, 'tags', None))