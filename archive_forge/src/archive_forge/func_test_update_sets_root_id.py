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
def test_update_sets_root_id(self):
    """Ensure tree root is set properly by update.

        Since empty trees don't have root_ids, but workingtrees do,
        an update of a checkout of revision 0 to a new revision,  should set
        the root id.
        """
    wt = self.make_branch_and_tree('tree')
    main_branch = wt.branch
    self.build_tree(['checkout/', 'tree/file'])
    checkout = main_branch.create_checkout('checkout')
    wt.add('file')
    a = wt.commit('A')
    self.assertEqual(0, checkout.update())
    self.assertPathExists('checkout/file')
    if wt.supports_setting_file_ids():
        self.assertEqual(wt.path2id(''), checkout.path2id(''))
        self.assertNotEqual(None, wt.path2id(''))