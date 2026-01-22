from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def make_failure(exc_value):
    """Raise ``exc_value`` and return the failure."""
    try:
        raise exc_value
    except:
        return Failure()