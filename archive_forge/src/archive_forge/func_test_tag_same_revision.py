from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_tag_same_revision(self):
    t = self.make_branch_and_tree('branch')
    t.commit(allow_pointless=True, message='initial commit', rev_id=b'first-revid')
    t.commit(allow_pointless=True, message='second commit', rev_id=b'second-revid')
    out, err = self.run_bzr('tag -rrevid:first-revid -d branch NEWTAG')
    out, err = self.run_bzr('tag -rrevid:first-revid -d branch NEWTAG')
    self.assertContainsRe(err, 'Tag NEWTAG already exists for that revision\\.')
    out, err = self.run_bzr('tag -rrevid:second-revid -d branch NEWTAG', retcode=3)
    self.assertContainsRe(err, 'Tag NEWTAG already exists\\.')