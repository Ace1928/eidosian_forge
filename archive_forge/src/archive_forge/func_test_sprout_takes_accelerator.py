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
def test_sprout_takes_accelerator(self):
    tree = self.make_branch_and_tree('source')
    self.build_tree(['source/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    rev2 = tree.commit('revision 2', allow_pointless=True)
    dir = tree.controldir
    target = self.sproutOrSkip(dir, self.get_url('target'), accelerator_tree=tree)
    self.assertEqual([rev2], target.open_workingtree().get_parent_ids())