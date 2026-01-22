import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_exits_nonzero_when_execution_errors(self):
    bytestream = io.BytesIO()
    stream = io.TextIOWrapper(bytestream, encoding='utf8')
    exc = self.assertRaises(SystemExit, run.main, argv=['progName', 'subunit.tests.test_run.TestSubunitTestRunner.ExitingTest'], stdout=stream)
    self.assertEqual(0, exc.args[0])