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
def test_list_files(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree(['dir/', 'file'])
    if supports_symlinks(self.test_dir):
        os.symlink('target', 'symlink')
    tree.lock_read()
    files = list(tree.list_files())
    tree.unlock()
    self.assertEqual(files.pop(0), ('dir', '?', 'directory', TreeDirectory()))
    self.assertEqual(files.pop(0), ('file', '?', 'file', TreeFile()))
    if supports_symlinks(self.test_dir):
        self.assertEqual(files.pop(0), ('symlink', '?', 'symlink', TreeLink()))