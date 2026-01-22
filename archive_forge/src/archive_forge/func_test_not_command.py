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
def test_not_command(self):
    client = unittest.TestResult()
    out = BytesIO()
    protocol = subunit.TestProtocolServer(client, stream=subunit.DiscardStream(), forward_stream=out)
    pipe = BytesIO(_b('success old mcdonald\n'))
    protocol.readFrom(pipe)
    self.assertEqual(client.testsRun, 0)
    self.assertEqual(_b(''), out.getvalue())