import codecs
import os
import time
from ... import errors, filters, osutils, rules
from ...controldir import ControlDir
from ...tests import UnavailableFeature, features
from ..conflicts import DuplicateEntry
from ..transform import build_tree
from . import TestCaseWithTransport
def test_build_tree_content_filtered_files_are_not_hardlinked(self):
    """build_tree will not hardlink files that have content filtering rules
        applied to them (but will still hardlink other files from the same tree
        if it can).
        """
    self.requireFeature(features.HardlinkFeature(self.test_dir))
    self.install_rot13_content_filter(b'file1')
    source = self.create_ab_tree()
    target = self.make_branch_and_tree('target')
    revision_tree = source.basis_tree()
    revision_tree.lock_read()
    self.addCleanup(revision_tree.unlock)
    build_tree(revision_tree, target, source, hardlink=True)
    target.lock_read()
    self.addCleanup(target.unlock)
    self.assertEqual([], list(target.iter_changes(revision_tree)))
    source_stat = os.stat('source/file1')
    target_stat = os.stat('target/file1')
    self.assertNotEqual(source_stat, target_stat)
    source_stat = os.stat('source/file2')
    target_stat = os.stat('target/file2')
    self.assertEqualStat(source_stat, target_stat)