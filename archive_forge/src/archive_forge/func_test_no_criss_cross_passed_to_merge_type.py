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
def test_no_criss_cross_passed_to_merge_type(self):

    class LCATreesMerger(LoggingMerger):
        supports_lca_trees = True
    merger = self.make_Merger(self.setup_simple_graph(), b'C-id')
    merger.merge_type = LCATreesMerger
    merge_obj = merger.make_merger()
    self.assertIsInstance(merge_obj, LCATreesMerger)
    self.assertFalse('lca_trees' in merge_obj.kwargs)