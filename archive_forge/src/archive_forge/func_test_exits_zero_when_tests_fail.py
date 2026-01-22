import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_exits_zero_when_tests_fail(self):
    bytestream = io.BytesIO()
    stream = io.TextIOWrapper(bytestream, encoding='utf8')
    try:
        self.assertEqual(None, run.main(argv=['progName', 'subunit.tests.test_run.TestSubunitTestRunner.FailingTest'], stdout=stream))
    except SystemExit:
        self.fail('SystemExit raised')
    self.assertThat(bytestream.getvalue(), StartsWith(_b('³')))