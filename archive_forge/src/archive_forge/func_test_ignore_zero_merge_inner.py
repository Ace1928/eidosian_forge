import contextlib
import os
from .. import branch as _mod_branch
from .. import conflicts, errors, memorytree
from .. import merge as _mod_merge
from .. import option
from .. import revision as _mod_revision
from .. import tests, transform
from ..bzr import inventory, knit, versionedfile
from ..bzr.conflicts import (ContentsConflict, DeletingParent, MissingParent,
from ..conflicts import ConflictList
from ..errors import NoCommits, UnrelatedBranches
from ..merge import _PlanMerge, merge_inner, transform_tree
from ..osutils import basename, file_kind, pathjoin
from ..workingtree import PointlessMerge, WorkingTree
from . import (TestCaseWithMemoryTransport, TestCaseWithTransport, features,
def test_ignore_zero_merge_inner(self):
    tree_a = self.make_branch_and_tree('a')
    tree_a.commit(message='hello')
    dir_b = tree_a.controldir.sprout('b')
    tree_b = dir_b.open_workingtree()
    tree_b.lock_write()
    self.addCleanup(tree_b.unlock)
    tree_a.commit(message='hello again')
    merge_inner(tree_b.branch, tree_a, tree_b.basis_tree(), this_tree=tree_b, ignore_zero=True)
    self.assertTrue('All changes applied successfully.\n' not in self.get_log())
    tree_b.revert()
    merge_inner(tree_b.branch, tree_a, tree_b.basis_tree(), this_tree=tree_b, ignore_zero=False)
    self.assertTrue('All changes applied successfully.\n' in self.get_log())