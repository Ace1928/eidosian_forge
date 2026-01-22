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
def test_update_revision(self):
    builder, tip, revids = self.make_diverged_master_branch()
    wt, master = self.make_checkout_and_master(builder, 'checkout', 'master', revids['4'], master_revid=tip, branch_revid=revids['2'])
    self.assertEqual(0, wt.update(revision=revids['1']))
    self.assertEqual(revids['1'], wt.last_revision())
    self.assertEqual(tip, wt.branch.last_revision())
    self.assertPathExists('checkout/file1')
    self.assertPathDoesNotExist('checkout/file4')
    self.assertPathDoesNotExist('checkout/file5')