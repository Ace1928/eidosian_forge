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
def test_merge_add_into_deleted_root(self):
    source = self.make_branch_and_tree('source', format='rich-root-pack')
    self.build_tree(['source/foo/'])
    source.add('foo', ids=b'foo-id')
    source.commit('Add foo')
    target = source.controldir.sprout('target').open_workingtree()
    subtree = target.extract('foo')
    subtree.commit('Delete root')
    self.build_tree(['source/bar'])
    source.add('bar', ids=b'bar-id')
    source.commit('Add bar')
    subtree.merge_from_branch(source.branch)