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
def test_nested_merge(self):
    self.knownFailure("iter_changes doesn't work with changes in nested trees")
    tree = self.make_branch_and_tree('tree', format='development-subtree')
    sub_tree = self.make_branch_and_tree('tree/sub-tree', format='development-subtree')
    sub_tree.set_root_id(b'sub-tree-root')
    self.build_tree_contents([('tree/sub-tree/file', b'text1')])
    sub_tree.add('file')
    sub_tree.commit('foo')
    tree.add_reference(sub_tree)
    tree.commit('set text to 1')
    tree2 = tree.controldir.sprout('tree2').open_workingtree()
    self.build_tree_contents([('tree2/sub-tree/file', b'text2')])
    tree2.commit('changed file text')
    tree.merge_from_branch(tree2.branch)
    self.assertFileEqual(b'text2', 'tree/sub-tree/file')