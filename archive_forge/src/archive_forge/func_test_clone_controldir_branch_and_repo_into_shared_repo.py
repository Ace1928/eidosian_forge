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
def test_clone_controldir_branch_and_repo_into_shared_repo(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    source = self.make_branch('source')
    tree.branch.repository.copy_content_into(source.repository)
    tree.branch.copy_content_into(source)
    try:
        shared_repo = self.make_repository('target', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('repository format does not support shared repositories')
    if not shared_repo._format.supports_nesting_repositories:
        raise TestNotApplicable('format does not support nesting repositories')
    dir = source.controldir
    target = dir.clone(self.get_url('target/child'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertRaises(errors.NoRepositoryPresent, target.open_repository)
    self.assertEqual(source.last_revision(), target.open_branch().last_revision())