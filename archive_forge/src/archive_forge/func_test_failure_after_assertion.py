from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_failure_after_assertion(self):
    deferred = defer.Deferred()
    self.assertThat(deferred, has_no_result())
    results = []
    deferred.addErrback(results.append)
    fail = make_failure(RuntimeError('arbitrary failure'))
    deferred.errback(fail)
    self.assertThat(results, Equals([fail]))