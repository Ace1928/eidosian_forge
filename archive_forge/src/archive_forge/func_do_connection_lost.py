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
def do_connection_lost(self, outcome, opening):
    self.protocol.lineReceived(_b('test old mcdonald\n'))
    self.protocol.lineReceived(_b('{} old mcdonald {}'.format(outcome, opening)))
    self.protocol.lostConnection()
    failure = subunit.RemoteError(_u("lost connection during %s report of test 'old mcdonald'") % outcome)
    self.assertEqual([('startTest', self.test), ('addError', self.test, failure), ('stopTest', self.test)], self.client._events)