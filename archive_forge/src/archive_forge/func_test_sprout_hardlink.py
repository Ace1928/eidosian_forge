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
def test_sprout_hardlink(self):
    real_os_link = getattr(os, 'link', None)
    if real_os_link is None:
        raise TestNotApplicable("This platform doesn't provide os.link")
    source = self.make_branch_and_tree('source')
    self.build_tree(['source/file'])
    source.add('file')
    source.commit('added file')

    def fake_link(source, target):
        raise OSError(errno.EPERM, 'Operation not permitted')
    os.link = fake_link
    try:
        try:
            source.controldir.sprout('target', accelerator_tree=source, hardlink=True)
        except errors.HardLinkNotSupported:
            pass
    finally:
        os.link = real_os_link