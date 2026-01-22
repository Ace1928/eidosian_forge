import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def compare_tree_fileids(self, branch, old_rev, new_rev):
    old_tree = self.branch.repository.revision_tree(old_rev)
    new_tree = self.branch.repository.revision_tree(new_rev)
    delta = new_tree.changes_from(old_tree)
    l2 = [change.file_id for change in delta.added] + [change.file_id for change in delta.renamed] + [change.file_id for change in delta.modified] + [change.file_id for change in delta.copied]
    return set(l2)