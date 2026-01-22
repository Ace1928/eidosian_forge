import os
from .. import check, osutils
from ..commit import PointlessCommit
from . import TestCaseWithTransport
from .features import SymlinkFeature
from .matchers import RevisionHistoryMatches
def test_merge_with_symlink(self):
    self.requireFeature(SymlinkFeature(self.test_dir))
    tree_a = self.make_branch_and_tree('tree_a')
    os.symlink('target', osutils.pathjoin('tree_a', 'link'))
    tree_a.add('link')
    tree_a.commit('added link')
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    self.build_tree(['tree_a/file'])
    tree_a.add('file')
    tree_a.commit('added file')
    self.build_tree(['tree_b/another_file'])
    tree_b.add('another_file')
    tree_b.commit('add another file')
    tree_b.merge_from_branch(tree_a.branch)
    tree_b.commit('merge')