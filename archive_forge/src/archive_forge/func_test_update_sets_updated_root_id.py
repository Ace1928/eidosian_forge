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
def test_update_sets_updated_root_id(self):
    wt = self.make_branch_and_tree('tree')
    if not wt.supports_setting_file_ids():
        self.assertRaises(SettingFileIdUnsupported, wt.set_root_id, 'first_root_id')
        return
    wt.set_root_id(b'first_root_id')
    self.assertEqual(b'first_root_id', wt.path2id(''))
    self.build_tree(['tree/file'])
    wt.add(['file'])
    wt.commit('first')
    co = wt.branch.create_checkout('checkout')
    wt.set_root_id(b'second_root_id')
    wt.commit('second')
    self.assertEqual(b'second_root_id', wt.path2id(''))
    self.assertEqual(0, co.update())
    self.assertEqual(b'second_root_id', co.path2id(''))