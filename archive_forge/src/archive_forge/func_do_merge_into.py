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
def do_merge_into(self, location, merge_as):
    """Helper for using MergeIntoMerger.

        :param location: location of directory to merge from, either the
            location of a branch or of a path inside a branch.
        :param merge_as: the path in a tree to add the new directory as.
        :returns: the conflicts from 'do_merge'.
        """
    with contextlib.ExitStack() as stack:
        wt, subdir_relpath = WorkingTree.open_containing(merge_as)
        stack.enter_context(wt.lock_write())
        branch_to_merge, subdir_to_merge = _mod_branch.Branch.open_containing(location)
        stack.enter_context(branch_to_merge.lock_read())
        other_tree = branch_to_merge.basis_tree()
        stack.enter_context(other_tree.lock_read())
        merger = _mod_merge.MergeIntoMerger(this_tree=wt, other_tree=other_tree, other_branch=branch_to_merge, target_subdir=subdir_relpath, source_subpath=subdir_to_merge)
        merger.set_base_revision(_mod_revision.NULL_REVISION, branch_to_merge)
        conflicts = merger.do_merge()
        merger.set_pending()
        return conflicts