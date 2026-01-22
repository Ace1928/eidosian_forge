from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_automatic_tag_name(self):

    def get_tag_name(branch, revid):
        return 'mytag'
    Branch.hooks.install_named_hook('automatic_tag_name', get_tag_name, 'get tag name')
    out, err = self.run_bzr('tag -d branch')
    self.assertContainsRe(err, 'Created tag mytag.')