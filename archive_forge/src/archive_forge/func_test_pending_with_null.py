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
def test_pending_with_null(self):
    """When base is forced to revno 0, parent_ids are set"""
    wt2 = self.test_unrelated()
    wt1 = WorkingTree.open('.')
    br1 = wt1.branch
    br1.fetch(wt2.branch)
    wt1.merge_from_branch(wt2.branch, wt2.last_revision(), b'null:')
    self.assertEqual([br1.last_revision(), wt2.branch.last_revision()], wt1.get_parent_ids())
    return (wt1, wt2.branch)