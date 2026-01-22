from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_failure_passes(self):
    fail = make_failure(RuntimeError('arbitrary failure'))
    deferred = defer.fail(fail)
    self.assertThat(self.match(Is(fail), deferred), Is(None))