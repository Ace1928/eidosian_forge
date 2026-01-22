import errno
import os
from io import StringIO
from ... import branch as _mod_branch
from ... import config, controldir, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...bzr import bzrdir
from ...bzr.conflicts import ConflictList, ContentsConflict, TextConflict
from ...bzr.inventory import Inventory
from ...bzr.workingtree import InventoryWorkingTree
from ...errors import PathsNotVersionedError, UnsupportedOperation
from ...mutabletree import MutableTree
from ...osutils import getcwd, pathjoin, supports_symlinks
from ...tree import TreeDirectory, TreeFile, TreeLink
from ...workingtree import SettingFileIdUnsupported, WorkingTree
from .. import TestNotApplicable, TestSkipped, features
from . import TestCaseWithWorkingTree
def test_update_remove_commit(self):
    """Update should remove revisions when the branch has removed
        some commits.

        We want to revert 4, so that strating with the
        make_diverged_master_branch() graph the final result should be
        equivalent to:

           1
           |           3 2
           | |        MB-5 | 4
           |/
           W

        And the changes in 4 have been removed from the WT.
        """
    builder, tip, revids = self.make_diverged_master_branch()
    wt, master = self.make_checkout_and_master(builder, 'checkout', 'master', revids['4'], master_revid=tip, branch_revid=revids['2'])
    old_tip = wt.branch.update()
    self.assertEqual(revids['2'], old_tip)
    self.assertEqual(0, wt.update(old_tip=old_tip))
    self.assertEqual(tip, wt.branch.last_revision())
    self.assertEqual([revids['5'], revids['2']], wt.get_parent_ids())