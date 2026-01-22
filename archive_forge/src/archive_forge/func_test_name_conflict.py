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
def test_name_conflict(self):
    """When the target directory name already exists a conflict is
        generated and the original directory is renamed to foo.moved.
        """
    dest_wt = self.setup_simple_branch('dest', ['dir/', 'dir/file.txt'])
    self.setup_simple_branch('src', ['README'])
    conflicts = self.do_merge_into('src', 'dest/dir')
    self.assertEqual(1, len(conflicts))
    dest_wt.lock_read()
    self.addCleanup(dest_wt.unlock)
    self.assertEqual([b'r1-dest', b'r1-src'], dest_wt.get_parent_ids())
    self.assertTreeEntriesEqual([('', b'dest-root-id'), ('dir', b'src-root-id'), ('dir.moved', b'dest-dir-id'), ('dir/README', b'src-README-id'), ('dir.moved/file.txt', b'dest-file.txt-id')], dest_wt)