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
def test_plan_merge_tail_triple_ancestors(self):
    self.add_rev(b'root', b'A', [], b'abc')
    self.add_rev(b'root', b'B', [b'A'], b'aBbc')
    self.add_rev(b'root', b'C', [b'A'], b'abCc')
    self.add_rev(b'root', b'D', [b'B'], b'DaBbc')
    self.add_rev(b'root', b'E', [b'B', b'C'], b'aBbCc')
    self.add_rev(b'root', b'F', [b'C'], b'abCcF')
    self.add_rev(b'root', b'G', [b'D', b'E'], b'DaBbCc')
    self.add_rev(b'root', b'H', [b'F', b'E'], b'aBbCcF')
    self.add_rev(b'root', b'Q', [b'E'], b'aBbCc')
    self.add_rev(b'root', b'I', [b'G', b'Q', b'H'], b'DaBbCcF')
    self.add_rev(b'root', b'J', [b'H', b'Q', b'G'], b'DaJbCcF')
    plan = self.plan_merge_vf.plan_merge(b'I', b'J')
    self.assertEqual([('unchanged', b'D\n'), ('unchanged', b'a\n'), ('killed-b', b'B\n'), ('new-b', b'J\n'), ('unchanged', b'b\n'), ('unchanged', b'C\n'), ('unchanged', b'c\n'), ('unchanged', b'F\n')], list(plan))