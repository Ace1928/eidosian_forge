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
def test_add_conflicts(self):
    tree = self.make_branch_and_tree('tree')
    try:
        tree.add_conflicts([TextConflict('path_a')])
    except UnsupportedOperation:
        raise TestSkipped('unsupported operation')
    self.assertEqual(ConflictList([TextConflict('path_a')]), tree.conflicts())
    tree.add_conflicts([TextConflict('path_a')])
    self.assertEqual(ConflictList([TextConflict('path_a')]), tree.conflicts())
    tree.add_conflicts([ContentsConflict('path_a')])
    self.assertEqual(ConflictList([ContentsConflict('path_a'), TextConflict('path_a')]), tree.conflicts())
    tree.add_conflicts([TextConflict('path_b')])
    self.assertEqual(ConflictList([ContentsConflict('path_a'), TextConflict('path_a'), TextConflict('path_b')]), tree.conflicts())