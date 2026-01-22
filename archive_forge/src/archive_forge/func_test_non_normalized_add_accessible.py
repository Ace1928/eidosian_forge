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
def test_non_normalized_add_accessible(self):
    try:
        self.build_tree(['å'])
    except UnicodeError:
        raise TestSkipped('Filesystem does not support unicode filenames')
    tree = self.make_branch_and_tree('.')
    orig = osutils.normalized_filename
    osutils.normalized_filename = osutils._accessible_normalized_filename
    try:
        tree.add(['å'])
        with tree.lock_read():
            self.assertEqual([('', 'directory'), ('å', 'file')], [(path, ie.kind) for path, ie in tree.iter_entries_by_dir()])
    finally:
        osutils.normalized_filename = orig