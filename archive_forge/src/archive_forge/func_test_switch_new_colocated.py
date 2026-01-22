import os
from breezy import branch, osutils, urlutils
from breezy.controldir import ControlDir
from breezy.directory_service import directories
from breezy.tests import TestCaseWithTransport, script
from breezy.tests.features import UnicodeFilenameFeature
from breezy.workingtree import WorkingTree
def test_switch_new_colocated(self):
    repo = self.make_repository('branch-1', format='development-colo')
    target_branch = repo.controldir.create_branch(name='foo')
    repo.controldir.set_branch_reference(target_branch)
    tree = repo.controldir.create_workingtree()
    self.build_tree(['branch-1/file-1', 'branch-1/file-2'])
    tree.add('file-1')
    revid1 = tree.commit('rev1')
    self.run_bzr(['switch', '-b', 'anotherbranch'], working_dir='branch-1')
    bzrdir = ControlDir.open('branch-1')
    self.assertEqual({b.name for b in bzrdir.list_branches()}, {'foo', 'anotherbranch'})
    self.assertEqual(bzrdir.open_branch().name, 'anotherbranch')
    self.assertEqual(bzrdir.open_branch().last_revision(), revid1)