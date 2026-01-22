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
def test_list_files_versioned_before_ignored(self):
    """A versioned file matching an ignore rule should not be ignored."""
    tree = self.make_branch_and_tree('.')
    self.build_tree(['foo.pyc'])
    self.build_tree_contents([('.bzrignore', b'foo.pyc')])
    tree.add('foo.pyc')
    tree.lock_read()
    files = sorted(list(tree.list_files()))
    tree.unlock()
    self.assertEqual(('.bzrignore', '?', 'file', None), (files[0][0], files[0][1], files[0][2], getattr(files[0][3], 'file_id', None)))
    self.assertEqual(('foo.pyc', 'V', 'file'), (files[1][0], files[1][1], files[1][2]))
    self.assertEqual(2, len(files))