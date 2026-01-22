from testtools.content import TracebackContent
from testtools.matchers import (
from ._helpers import NeedsTwistedTestCase
from testtools.twistedsupport import has_no_result, failed, succeeded
from twisted.internet import defer
from twisted.python.failure import Failure
def mismatches(description, details=None):
    """Match a ``Mismatch`` object."""
    if details is None:
        details = Equals({})
    matcher = MatchesDict({'description': description, 'details': details})

    def get_mismatch_info(mismatch):
        return {'description': mismatch.describe(), 'details': mismatch.get_details()}
    return AfterPreprocessing(get_mismatch_info, matcher)