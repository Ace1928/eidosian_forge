import io
import unittest
from testtools import PlaceHolder, TestCase
from testtools.compat import _b
from testtools.matchers import StartsWith
from testtools.testresult.doubles import StreamResult
import subunit
from subunit import run
from subunit.run import SubunitTestRunner
def test_list_includes_loader_errors(self):
    bytestream = io.BytesIO()
    runner = SubunitTestRunner(stream=bytestream)

    def list_test(test):
        return ([], [])

    class Loader:
        errors = ['failed import']
    loader = Loader()
    self.patch(run, 'list_test', list_test)
    exc = self.assertRaises(SystemExit, runner.list, None, loader=loader)
    self.assertEqual((2,), exc.args)