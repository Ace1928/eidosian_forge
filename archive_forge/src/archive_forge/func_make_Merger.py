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
def make_Merger(self, builder, other_revision_id, interesting_files=None):
    """Make a Merger object from a branch builder"""
    mem_tree = memorytree.MemoryTree.create_on_branch(builder.get_branch())
    mem_tree.lock_write()
    self.addCleanup(mem_tree.unlock)
    merger = _mod_merge.Merger.from_revision_ids(mem_tree, other_revision_id)
    merger.set_interesting_files(interesting_files)
    merger.merge_type = _mod_merge.Merge3Merger
    return merger