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
def test_set_last_revision(self):
    wt = self.make_branch_and_tree('source')
    if wt.branch.repository._format.supports_ghosts:
        wt.set_last_revision(b'A')
    wt.set_last_revision(b'null:')
    a = wt.commit('A', allow_pointless=True)
    self.assertEqual([a], wt.get_parent_ids())
    wt.set_last_revision(b'null:')
    self.assertEqual([], wt.get_parent_ids())
    if getattr(wt.branch, '_set_revision_history', None) is None:
        raise TestSkipped('Branch format does not permit arbitrary history')
    wt.branch._set_revision_history([a, b'B'])
    wt.set_last_revision(a)
    self.assertEqual([a], wt.get_parent_ids())
    self.assertRaises(errors.ReservedId, wt.set_last_revision, b'A:')