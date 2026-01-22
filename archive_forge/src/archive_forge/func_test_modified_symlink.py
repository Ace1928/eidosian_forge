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
def test_modified_symlink(self):
    self.requireFeature(features.SymlinkFeature(self.test_dir))
    wt = self.make_branch_and_tree('path')
    wt.lock_write()
    self.addCleanup(wt.unlock)
    os.symlink('bar', 'path/foo')
    wt.add(['foo'], ids=[b'foo-id'])
    wt.commit('add symlink', rev_id=b'A-id')
    os.remove('path/foo')
    os.symlink('baz', 'path/foo')
    wt.commit('foo => baz', rev_id=b'B-id')
    wt.set_last_revision(b'A-id')
    wt.branch.set_last_revision_info(1, b'A-id')
    wt.revert()
    wt.commit('C', rev_id=b'C-id')
    wt.merge_from_branch(wt.branch, b'B-id')
    self.assertEqual('baz', wt.get_symlink_target('foo'))
    wt.commit('E merges C & B', rev_id=b'E-id')
    os.remove('path/foo')
    os.symlink('bing', 'path/foo')
    wt.commit('F foo => bing', rev_id=b'F-id')
    wt.set_last_revision(b'B-id')
    wt.branch.set_last_revision_info(2, b'B-id')
    wt.revert()
    wt.merge_from_branch(wt.branch, b'C-id')
    wt.commit('D merges B & C', rev_id=b'D-id')
    conflicts = wt.merge_from_branch(wt.branch, to_revision=b'F-id')
    self.assertEqual(0, len(conflicts))
    self.assertEqual('bing', wt.get_symlink_target('foo'))