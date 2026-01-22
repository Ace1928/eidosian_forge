import errno
from stat import S_ISDIR
import breezy.branch
from breezy import controldir, errors, repository
from breezy import revision as _mod_revision
from breezy import transport, workingtree
from breezy.bzr import bzrdir
from breezy.bzr.remote import RemoteBzrDirFormat
from breezy.bzr.tests.per_bzrdir import TestCaseWithBzrDir
from breezy.tests import TestNotApplicable, TestSkipped
from breezy.transport import FileExists
from breezy.transport.local import LocalTransport
def test_clone_bzrdir_branch_and_repo(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['commit_tree/foo'])
    tree.add('foo')
    tree.commit('revision 1')
    source = self.make_branch('source')
    tree.branch.repository.copy_content_into(source.repository)
    tree.branch.copy_content_into(source)
    dir = source.controldir
    target = dir.clone(self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport, ['./.bzr/basis-inventory-cache', './.bzr/checkout/stat-cache', './.bzr/merge-hashes', './.bzr/repository', './.bzr/stat-cache'])
    self.assertRepositoryHasSameItems(tree.branch.repository, target.open_repository())