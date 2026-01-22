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
def test_subtract_plans(self):
    old_plan = [('unchanged', b'a\n'), ('new-a', b'b\n'), ('killed-a', b'c\n'), ('new-b', b'd\n'), ('new-b', b'e\n'), ('killed-b', b'f\n'), ('killed-b', b'g\n')]
    new_plan = [('unchanged', b'a\n'), ('new-a', b'b\n'), ('killed-a', b'c\n'), ('new-b', b'd\n'), ('new-b', b'h\n'), ('killed-b', b'f\n'), ('killed-b', b'i\n')]
    subtracted_plan = [('unchanged', b'a\n'), ('new-a', b'b\n'), ('killed-a', b'c\n'), ('new-b', b'h\n'), ('unchanged', b'f\n'), ('killed-b', b'i\n')]
    self.assertEqual(subtracted_plan, list(_PlanMerge._subtract_plans(old_plan, new_plan)))