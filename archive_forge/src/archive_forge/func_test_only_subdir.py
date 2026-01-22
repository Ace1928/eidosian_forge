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
def test_only_subdir(self):
    """When the location points to just part of a tree, merge just that
        subtree.
        """
    dest_wt = self.setup_simple_branch('dest')
    self.setup_simple_branch('src', ['hello.txt', 'dir/', 'dir/foo.c'])
    self.do_merge_into('src/dir', 'dest/dir')
    dest_wt.lock_read()
    self.addCleanup(dest_wt.unlock)
    self.assertEqual([b'r1-dest'], dest_wt.get_parent_ids())
    self.assertTreeEntriesEqual([('', b'dest-root-id'), ('dir', b'src-dir-id'), ('dir/foo.c', b'src-foo.c-id')], dest_wt)