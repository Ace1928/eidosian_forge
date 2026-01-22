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
def test_newdir_with_repeat_roots(self):
    """If the file-id of the dir to be merged already exists a new ID will
        be allocated to let the merge happen.
        """
    project_wt, lib_wt = self.setup_two_branches(custom_root_ids=False)
    root_id = project_wt.path2id('')
    self.do_merge_into('lib1', 'project/lib1')
    project_wt.lock_read()
    self.addCleanup(project_wt.unlock)
    self.assertEqual([b'r1-project', b'r1-lib1'], project_wt.get_parent_ids())
    new_lib1_id = project_wt.path2id('lib1')
    self.assertNotEqual(None, new_lib1_id)
    self.assertTreeEntriesEqual([('', root_id), ('README', b'project-README-id'), ('dir', b'project-dir-id'), ('lib1', new_lib1_id), ('dir/file.c', b'project-file.c-id'), ('lib1/Makefile', b'lib1-Makefile-id'), ('lib1/README', b'lib1-README-id'), ('lib1/foo.c', b'lib1-foo.c-id')], project_wt)