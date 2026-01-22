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
def test_cache_trees_with_revision_ids_no_revision_id(self):
    merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
    original_cache = dict(merger._cached_trees)
    tree = self.make_branch_and_memory_tree('tree')
    merger.cache_trees_with_revision_ids([tree])
    self.assertEqual(original_cache, merger._cached_trees)