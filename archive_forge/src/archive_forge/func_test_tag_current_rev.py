from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_tag_current_rev(self):
    t = self.make_branch_and_tree('branch')
    t.commit(allow_pointless=True, message='initial commit', rev_id=b'first-revid')
    out, err = self.run_bzr('tag -d branch NEWTAG')
    self.assertContainsRe(err, 'Created tag NEWTAG.')
    self.assertEqual(t.branch.tags.get_tag_dict(), dict(NEWTAG=b'first-revid'))
    self.run_bzr('tag -d branch tag2 -r1')
    self.assertEqual(t.branch.tags.lookup_tag('tag2'), b'first-revid')
    self.run_bzr(['tag', '-d', 'branch', 'tag3', '-rrevid:first-revid'])
    self.assertEqual(t.branch.tags.lookup_tag('tag3'), b'first-revid')
    out, err = self.run_bzr('tag --delete -d branch tag2')
    out, err = self.run_bzr('tag -d branch NEWTAG -r0', retcode=3)
    self.assertContainsRe(err, 'Tag NEWTAG already exists\\.')
    out, err = self.run_bzr('tag -d branch NEWTAG --force -r0')
    self.assertEqual('Updated tag NEWTAG.\n', err)