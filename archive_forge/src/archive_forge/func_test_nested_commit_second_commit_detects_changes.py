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
def test_nested_commit_second_commit_detects_changes(self):
    """Commit with a nested tree picks up the correct child revid."""
    tree = self.make_branch_and_tree('.')
    if not tree.supports_tree_reference():
        return
    self.knownFailure("nested trees don't work well with iter_changes")
    subtree = self.make_branch_and_tree('subtree')
    tree.add(['subtree'])
    self.build_tree(['subtree/file'])
    subtree.add(['file'])
    rev_id = tree.commit('added reference', allow_pointless=False)
    tree.get_reference_revision('subtree')
    child_revid = subtree.last_revision()
    self.build_tree_contents([('subtree/file', b'new-content')])
    rev_id2 = tree.commit('changed subtree only', allow_pointless=False)
    self.assertNotEqual(None, subtree.last_revision())
    self.assertNotEqual(child_revid, subtree.last_revision())
    basis = tree.basis_tree()
    basis.lock_read()
    self.addCleanup(basis.unlock)
    self.assertEqual(subtree.last_revision(), basis.get_reference_revision('subtree'))
    self.assertNotEqual(rev_id, rev_id2)