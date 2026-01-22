from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_different_failure_fails(self):
    fail = make_failure(RuntimeError('arbitrary failure'))
    deferred = defer.fail(fail)
    matcher = Is(None)
    mismatch = matcher.match(fail)
    self.assertThat(self.match(matcher, deferred), mismatches(Equals(mismatch.describe()), Equals(mismatch.get_details())))