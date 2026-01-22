from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_succeeded_does_no_match(self):
    result = object()
    deferred = defer.succeed(result)
    mismatch = self.match(deferred)
    self.assertThat(mismatch, mismatches(Equals('No result expected on %r, found %r instead' % (deferred, result))))