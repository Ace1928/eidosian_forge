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
def setup_plan_merge_uncommitted(self):
    self.add_version((b'root', b'A'), [], b'abc')
    self.add_uncommitted_version((b'root', b'B:'), [(b'root', b'A')], b'acehg')
    self.add_uncommitted_version((b'root', b'C:'), [(b'root', b'A')], b'fabg')
    return _PlanMerge(b'B:', b'C:', self.plan_merge_vf, (b'root',))