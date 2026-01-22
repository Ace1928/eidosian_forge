import sys
from unittest import TestLoader, TestSuite
from breezy.tests import TestCaseWithTransport
def test_mirror_revs_into(self):
    self.make_branch_and_tree('source')
    self.make_branch_and_tree('dest')
    out, err = self.run_bzr('mirror-revs-into source dest')
    self.assertEqual(out, '')
    self.assertEqual(err, '')