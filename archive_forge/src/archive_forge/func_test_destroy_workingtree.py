import breezy.branch
from breezy import branch as _mod_branch
from breezy import check, controldir, errors, gpg, osutils
from breezy import repository as _mod_repository
from breezy import revision as _mod_revision
from breezy import transport, ui, urlutils, workingtree
from breezy.bzr import bzrdir as _mod_bzrdir
from breezy.bzr.remote import (RemoteBzrDir, RemoteBzrDirFormat,
from breezy.tests import (ChrootedTestCase, TestNotApplicable, TestSkipped,
from breezy.tests.per_controldir import TestCaseWithControlDir
from breezy.transport.local import LocalTransport
from breezy.ui import CannedInputUIFactory
def test_destroy_workingtree(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add('file')
    tree.commit('first commit')
    bzrdir = tree.controldir
    try:
        bzrdir.destroy_workingtree()
    except errors.UnsupportedOperation:
        raise TestSkipped('Format does not support destroying tree')
    self.assertPathDoesNotExist('tree/file')
    self.assertRaises(errors.NoWorkingTree, bzrdir.open_workingtree)
    bzrdir.create_workingtree()
    self.assertPathExists('tree/file')
    bzrdir.destroy_workingtree_metadata()
    self.assertPathExists('tree/file')
    self.assertRaises(errors.NoWorkingTree, bzrdir.open_workingtree)