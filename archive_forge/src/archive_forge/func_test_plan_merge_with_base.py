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
def test_plan_merge_with_base(self):
    self.setup_merge_with_base()
    plan = self.plan_merge_vf.plan_merge(b'THIS', b'OTHER', b'BASE')
    self.assertEqual([('unchanged', b'a\n'), ('new-b', b'f\n'), ('unchanged', b'b\n'), ('killed-b', b'c\n'), ('new-a', b'd\n')], list(plan))