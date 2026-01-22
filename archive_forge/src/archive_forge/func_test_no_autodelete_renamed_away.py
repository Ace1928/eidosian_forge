import os
from breezy import branch, conflicts, controldir, errors, mutabletree, osutils
from breezy import revision as _mod_revision
from breezy import tests
from breezy import transport as _mod_transport
from breezy import ui
from breezy.commit import CannotCommitSelectedFileMerge, PointlessCommit
from breezy.tests.matchers import HasPathRelations
from breezy.tests.per_workingtree import TestCaseWithWorkingTree
from breezy.tests.testui import ProgressRecordingUIFactory
def test_no_autodelete_renamed_away(self):
    tree_a = self.make_branch_and_tree('a')
    self.build_tree(['a/dir/', 'a/dir/f1', 'a/dir/f2', 'a/dir2/'])
    tree_a.add(['dir', 'dir/f1', 'dir/f2', 'dir2'])
    rev_id1 = tree_a.commit('init')
    revtree = tree_a.branch.repository.revision_tree(rev_id1)
    tree_a.rename_one('dir/f1', 'dir2/a')
    osutils.rmtree('a/dir')
    tree_a.commit('autoremoved')
    self.assertThat(tree_a, HasPathRelations(revtree, [('', ''), ('dir2/', 'dir2/'), ('dir2/a', 'dir/f1')]))