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
def test_plan_lca_merge_with_criss_cross(self):
    self.add_version((b'root', b'ROOT'), [], b'abc')
    self.add_version((b'root', b'REV1'), [(b'root', b'ROOT')], b'abcd')
    self.add_version((b'root', b'REV2'), [(b'root', b'ROOT')], b'abce')
    self.add_version((b'root', b'LCA1'), [(b'root', b'REV1'), (b'root', b'REV2')], b'abcd')
    self.add_version((b'root', b'LCA2'), [(b'root', b'REV1'), (b'root', b'REV2')], b'fabce')
    plan = self.plan_merge_vf.plan_lca_merge(b'LCA1', b'LCA2')
    self.assertEqual([('new-b', b'f\n'), ('unchanged', b'a\n'), ('unchanged', b'b\n'), ('unchanged', b'c\n'), ('conflicted-a', b'd\n'), ('conflicted-b', b'e\n')], list(plan))