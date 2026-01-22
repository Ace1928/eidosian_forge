from breezy.tests import TestCaseWithTransport
def test_parent_missing(self):
    wt = self.make_branch_and_tree('.')
    out, err = self.run_bzr('resolve-location :parent', retcode=3)
    self.assertEqual(out, '')
    self.assertEqual(err, 'brz: ERROR: No parent location assigned.\n')