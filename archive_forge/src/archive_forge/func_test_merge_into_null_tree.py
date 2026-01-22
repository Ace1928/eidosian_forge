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
def test_merge_into_null_tree(self):
    wt = self.make_branch_and_tree('tree')
    null_tree = wt.basis_tree()
    self.build_tree(['tree/file'])
    wt.add('file')
    wt.commit('tree with root')
    merger = _mod_merge.Merge3Merger(null_tree, null_tree, null_tree, wt, this_branch=wt.branch, do_merge=False)
    with merger.make_preview_transform() as tt:
        self.assertEqual([], tt.find_raw_conflicts())
        preview = tt.get_preview_tree()
        self.assertEqual(wt.path2id(''), preview.path2id(''))