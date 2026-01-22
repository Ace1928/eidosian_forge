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
def test_sprout_bzrdir_repository(self):
    tree = self.make_branch_and_tree('commit_tree')
    self.build_tree(['foo'], transport=tree.controldir.transport.clone('..'))
    tree.add('foo')
    tree.commit('revision 1', rev_id=b'1')
    dir = self.make_controldir('source')
    repo = dir.create_repository()
    repo.fetch(tree.branch.repository)
    self.assertTrue(repo.has_revision(b'1'))
    try:
        self.assertTrue(_mod_revision.is_null(dir.open_branch().last_revision()))
    except errors.NotBranchError:
        pass
    target = dir.sprout(self.get_url('target'))
    self.assertNotEqual(dir.transport.base, target.transport.base)
    self.assertDirectoriesEqual(dir.root_transport, target.root_transport, ['./.bzr/branch', './.bzr/checkout', './.bzr/inventory', './.bzr/parent', './.bzr/repository/inventory.knit'])
    try:
        local_inventory = dir.transport.local_abspath('inventory')
    except errors.NotLocalUrl:
        return
    try:
        with open(local_inventory, 'rb') as inventory_f:
            self.assertContainsRe(inventory_f.read(), b'<inventory format="5">\n</inventory>\n')
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise