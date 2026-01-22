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
def test_merge_with_missing(self):
    tree_a = self.make_branch_and_tree('tree_a')
    self.build_tree_contents([('tree_a/file', b'content_1')])
    tree_a.add('file')
    tree_a.commit('commit base')
    base_tree = tree_a.branch.repository.revision_tree(tree_a.last_revision())
    tree_b = tree_a.controldir.sprout('tree_b').open_workingtree()
    self.build_tree_contents([('tree_a/file', b'content_2')])
    tree_a.commit('commit other')
    other_tree = tree_a.basis_tree()
    os.unlink('tree_b/file')
    merge_inner(tree_b.branch, other_tree, base_tree, this_tree=tree_b)