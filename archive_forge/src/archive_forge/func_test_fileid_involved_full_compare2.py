import sys
import time
from breezy import errors
from breezy import revision as _mod_revision
from breezy import tests, transform
from breezy.bzr import inventory, remote
from breezy.bzr.tests.per_repository_vf import (
from breezy.tests.scenarios import load_tests_apply_scenarios
def test_fileid_involved_full_compare2(self):
    self.branch.lock_read()
    self.addCleanup(self.branch.unlock)
    graph = self.branch.repository.get_graph()
    history = list(graph.iter_lefthand_ancestry(self.branch.last_revision(), [_mod_revision.NULL_REVISION]))
    history.reverse()
    old_rev = history[0]
    new_rev = history[1]
    unique_revs = graph.find_unique_ancestors(new_rev, [old_rev])
    l1 = self.branch.repository.fileids_altered_by_revision_ids(unique_revs)
    l1 = set(l1.keys())
    l2 = self.compare_tree_fileids(self.branch, old_rev, new_rev)
    self.assertNotEqual(l2, l1)
    self.assertSubset(l2, l1)