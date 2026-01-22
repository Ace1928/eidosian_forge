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
def make_merge_conflicts(self):
    from breezy.merge import merge_inner
    tree = self.make_branch_and_tree('mine')
    with open('mine/bloo', 'wb') as f:
        f.write(b'one')
    with open('mine/blo', 'wb') as f:
        f.write(b'on')
    tree.add(['bloo', 'blo'])
    tree.commit('blah', allow_pointless=False)
    base = tree.branch.repository.revision_tree(tree.last_revision())
    controldir.ControlDir.open('mine').sprout('other')
    with open('other/bloo', 'wb') as f:
        f.write(b'two')
    othertree = WorkingTree.open('other')
    othertree.commit('blah', allow_pointless=False)
    with open('mine/bloo', 'wb') as f:
        f.write(b'three')
    tree.commit('blah', allow_pointless=False)
    merge_inner(tree.branch, othertree, base, this_tree=tree)
    return tree