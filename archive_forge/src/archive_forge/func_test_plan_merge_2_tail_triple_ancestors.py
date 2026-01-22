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
def test_plan_merge_2_tail_triple_ancestors(self):
    self.add_rev(b'root', b'A', [], b'abc')
    self.add_rev(b'root', b'B', [], b'def')
    self.add_rev(b'root', b'D', [b'A'], b'Dabc')
    self.add_rev(b'root', b'E', [b'A', b'B'], b'abcdef')
    self.add_rev(b'root', b'F', [b'B'], b'defF')
    self.add_rev(b'root', b'G', [b'D', b'E'], b'Dabcdef')
    self.add_rev(b'root', b'H', [b'F', b'E'], b'abcdefF')
    self.add_rev(b'root', b'Q', [b'E'], b'abcdef')
    self.add_rev(b'root', b'I', [b'G', b'Q', b'H'], b'DabcdefF')
    self.add_rev(b'root', b'J', [b'H', b'Q', b'G'], b'DabcdJfF')
    plan = self.plan_merge_vf.plan_merge(b'I', b'J')
    self.assertEqual([('unchanged', b'D\n'), ('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('unchanged', b'd\n'), ('killed-b', b'e\n'), ('new-b', b'J\n'), ('unchanged', b'f\n'), ('unchanged', b'F\n')], list(plan))