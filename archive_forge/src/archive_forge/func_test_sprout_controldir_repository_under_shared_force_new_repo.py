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
def test_sprout_controldir_repository_under_shared_force_new_repo(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    rev1 = tree.commit('revision 1')
    tree.controldir.open_branch().generate_revision_history(_mod_revision.NULL_REVISION)
    tree.set_parent_trees([])
    tree.commit('revision 2')
    source = self.make_repository('source')
    tree.branch.repository.copy_content_into(source)
    dir = source.controldir
    try:
        shared_repo = self.make_repository('target', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('format does not support shared repositories')
    target = dir.sprout(self.get_url('target/child'), force_new_repo=True)
    self.assertNotEqual(dir.control_transport.base, target.control_transport.base)
    self.assertFalse(shared_repo.has_revision(rev1))