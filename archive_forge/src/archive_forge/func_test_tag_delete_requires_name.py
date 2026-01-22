from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_tag_delete_requires_name(self):
    out, err = self.run_bzr('tag -d branch', retcode=3)
    self.assertContainsRe(err, 'Please specify a tag name\\.')