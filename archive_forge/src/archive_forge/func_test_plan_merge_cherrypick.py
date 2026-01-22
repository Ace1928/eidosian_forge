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
def test_plan_merge_cherrypick(self):
    self.add_rev(b'root', b'A', [], b'abc')
    self.add_rev(b'root', b'B', [b'A'], b'abcde')
    self.add_rev(b'root', b'C', [b'A'], b'abcefg')
    self.add_rev(b'root', b'D', [b'A', b'B', b'C'], b'abcdegh')
    my_plan = _PlanMerge(b'B', b'D', self.plan_merge_vf, (b'root',))
    self.assertEqual([('new-b', b'a\n'), ('new-b', b'b\n'), ('new-b', b'c\n'), ('new-b', b'd\n'), ('new-b', b'e\n'), ('new-b', b'g\n'), ('new-b', b'h\n')], list(my_plan.plan_merge()))