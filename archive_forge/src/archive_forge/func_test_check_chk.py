import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_check_chk(self):
    out, err = self.run_bzr('check-chk')
    self.assertEqual(out, '')
    self.assertEqual(err, '')