from testtools import (
from testtools.matchers import HasLength, MatchesException, Is, Raises
from testtools.testresult.doubles import ExtendedTestResult
from testtools.tests.helpers import FullStackRunTest
def make_case(self):

    class Case(TestCase):

        def test(self):
            pass
    return Case('test')