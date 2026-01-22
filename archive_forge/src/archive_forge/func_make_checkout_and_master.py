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
def make_checkout_and_master(self, builder, wt_path, master_path, wt_revid, master_revid=None, branch_revid=None):
    """Build a lightweight checkout and its master branch."""
    if master_revid is None:
        master_revid = wt_revid
    if branch_revid is None:
        branch_revid = master_revid
    final_branch = builder.get_branch()
    master = final_branch.controldir.sprout(master_path, master_revid).open_branch()
    wt = self.make_branch_and_tree(wt_path)
    wt.pull(final_branch, stop_revision=wt_revid)
    wt.branch.pull(final_branch, stop_revision=branch_revid, overwrite=True)
    try:
        wt.branch.bind(master)
    except _mod_branch.BindingUnsupported:
        raise TestNotApplicable("Can't bind %s" % wt.branch._format.__class__)
    return (wt, master)