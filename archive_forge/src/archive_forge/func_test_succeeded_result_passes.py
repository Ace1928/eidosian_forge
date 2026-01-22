from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def test_succeeded_result_passes(self):
    result = object()
    deferred = defer.succeed(result)
    self.assertThat(self.match(Is(result), deferred), Is(None))