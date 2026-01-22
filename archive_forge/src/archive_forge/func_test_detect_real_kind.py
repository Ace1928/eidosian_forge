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
def test_detect_real_kind(self):
    tree = self.make_branch_and_tree('.')
    tree.lock_write()
    self.addCleanup(tree.unlock)
    self.build_tree(['file', 'directory/'])
    names = ['file', 'directory']
    if supports_symlinks(self.test_dir):
        os.symlink('target', 'symlink')
        names.append('symlink')
    tree.add(names)
    for n in names:
        actual_kind = tree.kind(n)
        self.assertEqual(n, actual_kind)
    os.rename(names[0], 'tmp')
    for i in range(1, len(names)):
        os.rename(names[i], names[i - 1])
    os.rename('tmp', names[-1])
    for i in range(len(names)):
        actual_kind = tree.kind(names[i - 1])
        expected_kind = names[i]
        self.assertEqual(expected_kind, actual_kind)