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
def test_supports_executable(self):
    self.build_tree(['filename'])
    tree = self.make_branch_and_tree('.')
    tree.add('filename')
    self.assertIsInstance(tree._supports_executable(), bool)
    if tree._supports_executable():
        tree.lock_read()
        try:
            self.assertFalse(tree.is_executable('filename'))
        finally:
            tree.unlock()
        os.chmod('filename', 493)
        self.addCleanup(tree.lock_read().unlock)
        self.assertTrue(tree.is_executable('filename'))
    else:
        self.addCleanup(tree.lock_read().unlock)
        self.assertFalse(tree.is_executable('filename'))