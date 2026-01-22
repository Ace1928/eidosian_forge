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
def test_clone_controldir_repository_under_shared(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['foo'], transport=tree.controldir.transport.clone('..'))
    tree.add('foo')
    rev1 = tree.commit('revision 1')
    dir = self.make_controldir('source')
    repo = dir.create_repository()
    if not repo._format.supports_nesting_repositories:
        raise TestNotApplicable('repository format does not support nesting')
    repo.fetch(tree.branch.repository)
    self.assertTrue(repo.has_revision(rev1))
    try:
        self.make_repository('target', shared=True)
    except errors.IncompatibleFormat:
        raise TestNotApplicable('repository format does not support shared repositories')
    target = dir.clone(self.get_url('target/child'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertRaises(errors.NoRepositoryPresent, target.open_repository)