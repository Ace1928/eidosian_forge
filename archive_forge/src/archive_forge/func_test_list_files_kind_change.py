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
def test_list_files_kind_change(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/filename'])
    tree.add('filename')
    os.unlink('tree/filename')
    self.build_tree(['tree/filename/'])
    tree.lock_read()
    self.addCleanup(tree.unlock)
    result = list(tree.list_files())
    self.assertEqual(1, len(result))
    if tree.has_versioned_directories():
        self.assertEqual(('filename', 'V', 'directory'), (result[0][0], result[0][1], result[0][2]))
    else:
        self.assertEqual(('filename', '?', 'directory'), (result[0][0], result[0][1], result[0][2]))