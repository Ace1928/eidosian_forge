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
def test_plan_merge_with_delete_and_change(self):
    self.add_rev(b'root', b'C', [], b'a')
    self.add_rev(b'root', b'A', [b'C'], b'b')
    self.add_rev(b'root', b'B', [b'C'], b'')
    plan = self.plan_merge_vf.plan_merge(b'A', b'B')
    self.assertEqual([('killed-both', b'a\n'), ('new-a', b'b\n')], list(plan))