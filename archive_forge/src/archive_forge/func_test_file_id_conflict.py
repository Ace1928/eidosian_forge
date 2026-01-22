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
def test_file_id_conflict(self):
    """A conflict is generated if the merge-into adds a file (or other
        inventory entry) with a file-id that already exists in the target tree.
        """
    self.setup_simple_branch('dest', ['file.txt'])
    src_wt = self.make_branch_and_tree('src')
    self.build_tree(['src/README'])
    src_wt.add(['README'], ids=[b'dest-file.txt-id'])
    src_wt.commit('Rev 1 of src.', rev_id=b'r1-src')
    conflicts = self.do_merge_into('src', 'dest/dir')
    self.assertEqual(1, len(conflicts))