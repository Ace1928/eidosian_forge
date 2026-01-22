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
def test_add_expected_failure(self):
    """Test addExpectedFailure on a TestProtocolClient."""
    self.protocol.addExpectedFailure(self.test, subunit.RemoteError(_u('phwoar crikey')))
    self.assertThat(self.io.getvalue(), MatchesAny(Equals(_b(('xfail: %s [\n' + _remote_exception_str + ': phwoar crikey\n]\n') % self.test.id())), Equals(_b(('xfail: %s [\n' + _remote_exception_repr + ': phwoar crikey\n]\n') % self.test.id()))))