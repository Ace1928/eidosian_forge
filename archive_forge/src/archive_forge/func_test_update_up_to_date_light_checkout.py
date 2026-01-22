import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_up_to_date_light_checkout(self):
    self.make_branch_and_tree('branch')
    self.run_bzr('checkout --lightweight branch checkout')
    out, err = self.run_bzr('update checkout')
    self.assertEqual('Tree is up to date at revision 0 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
    self.assertEqual('', out)