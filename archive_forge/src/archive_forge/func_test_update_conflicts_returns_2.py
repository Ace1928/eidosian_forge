import os
from breezy import branch, osutils, tests, workingtree
from breezy.bzr import bzrdir
from breezy.tests.script import ScriptRunner
def test_update_conflicts_returns_2(self):
    self.make_branch_and_tree('branch')
    self.run_bzr('checkout --lightweight branch checkout')
    self.build_tree(['checkout/file'])
    self.run_bzr('add checkout/file')
    self.run_bzr('commit -m add-file checkout')
    self.run_bzr('checkout --lightweight branch checkout2')
    with open('checkout/file', 'w') as a_file:
        a_file.write('Foo')
    self.run_bzr('commit -m checnge-file checkout')
    with open('checkout2/file', 'w') as a_file:
        a_file.write('Bar')
    out, err = self.run_bzr('update checkout2', retcode=1)
    self.assertEqualDiff(' M  file\nText conflict in file\n1 conflicts encountered.\nUpdated to revision 2 of branch %s\n' % osutils.pathjoin(self.test_dir, 'branch'), err)
    self.assertEqual('', out)