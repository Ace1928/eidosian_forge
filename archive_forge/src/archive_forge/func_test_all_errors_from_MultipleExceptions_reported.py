from doctest import ELLIPSIS
from pprint import pformat
import sys
import _thread
import unittest
from testtools import (
from testtools.compat import (
from testtools.content import (
from testtools.matchers import (
from testtools.testcase import (
from testtools.testresult.doubles import (
from testtools.tests.helpers import (
from testtools.tests.samplecases import (
def test_all_errors_from_MultipleExceptions_reported(self):

    def raise_many(ignored):
        try:
            1 / 0
        except Exception:
            exc_info1 = sys.exc_info()
        try:
            1 / 0
        except Exception:
            exc_info2 = sys.exc_info()
        raise MultipleExceptions(exc_info1, exc_info2)
    test = make_test_case(self.getUniqueString(), cleanups=[raise_many])
    log = []
    test.run(ExtendedTestResult(log))
    self.assertThat(log, MatchesEvents(('startTest', test), ('addError', test, {'traceback': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError'])), 'traceback-1': AsText(ContainsAll(['Traceback (most recent call last):', 'ZeroDivisionError']))}), ('stopTest', test)))