import contextlib
import errno
import os
import sys
from typing import TYPE_CHECKING, Optional, Tuple
import breezy
from .lazy_import import lazy_import
import stat
from breezy import (
from . import errors, mutabletree, osutils
from . import revision as _mod_revision
from .controldir import (ControlComponent, ControlComponentFormat,
from .i18n import gettext
from .symbol_versioning import deprecated_in, deprecated_method
from .trace import mutter, note
from .transport import NoSuchFile
def merge_from_branch(self, branch, to_revision=None, from_revision=None, merge_type=None, force=False):
    """Merge from a branch into this working tree.

        Args:
          branch: The branch to merge from.
          to_revision: If non-None, the merge will merge to to_revision,
            but not beyond it. to_revision does not need to be in the history
            of the branch when it is supplied. If None, to_revision defaults to
            branch.last_revision().
        """
    from .merge import Merge3Merger, Merger
    with self.lock_write():
        merger = Merger(self.branch, this_tree=self)
        if not force and self.has_changes():
            raise errors.UncommittedChanges(self)
        if to_revision is None:
            to_revision = branch.last_revision()
        merger.other_rev_id = to_revision
        if _mod_revision.is_null(merger.other_rev_id):
            raise errors.NoCommits(branch)
        self.branch.fetch(branch, stop_revision=merger.other_rev_id)
        merger.other_basis = merger.other_rev_id
        merger.other_tree = self.branch.repository.revision_tree(merger.other_rev_id)
        merger.other_branch = branch
        if from_revision is None:
            merger.find_base()
        else:
            merger.set_base_revision(from_revision, branch)
        if merger.base_rev_id == merger.other_rev_id:
            raise PointlessMerge()
        merger.backup_files = False
        if merge_type is None:
            merger.merge_type = Merge3Merger
        else:
            merger.merge_type = merge_type
        merger.set_interesting_files(None)
        merger.show_base = False
        merger.reprocess = False
        conflicts = merger.do_merge()
        merger.set_pending()
        return conflicts