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
def test_rename_dirs(self):
    """Test renaming directories and the files within them."""
    wt = self.make_branch_and_tree('.')
    b = wt.branch
    self.build_tree(['dir/', 'dir/sub/', 'dir/sub/file'])
    wt.add(['dir', 'dir/sub', 'dir/sub/file'])
    wt.commit('create initial state')
    revid = b.last_revision()
    self.log('first revision_id is {%s}' % revid)
    tree = b.repository.revision_tree(revid)
    self.log('contents of tree: %r' % list(tree.iter_entries_by_dir()))
    self.check_tree_shape(tree, ['dir/', 'dir/sub/', 'dir/sub/file'])
    wt.rename_one('dir', 'newdir')
    wt.lock_read()
    self.check_tree_shape(wt, ['newdir/', 'newdir/sub/', 'newdir/sub/file'])
    wt.unlock()
    wt.rename_one('newdir/sub', 'newdir/newsub')
    wt.lock_read()
    self.check_tree_shape(wt, ['newdir/', 'newdir/newsub/', 'newdir/newsub/file'])
    wt.unlock()