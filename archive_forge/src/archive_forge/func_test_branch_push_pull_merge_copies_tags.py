from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_branch_push_pull_merge_copies_tags(self):
    t = self.make_branch_and_tree('branch1')
    t.commit(allow_pointless=True, message='initial commit', rev_id=b'first-revid')
    b1 = t.branch
    b1.tags.set_tag('tag1', b'first-revid')
    self.run_bzr('branch branch1 branch2')
    b2 = Branch.open('branch2')
    self.assertEqual(b2.tags.lookup_tag('tag1'), b'first-revid')
    b1.tags.set_tag('tag2', b'twa')
    self.run_bzr('pull -d branch2 branch1')
    self.assertEqual(b2.tags.lookup_tag('tag2'), b'twa')
    b1.tags.set_tag('tag3', b'san')
    self.run_bzr('push -d branch1 branch2')
    self.assertEqual(b2.tags.lookup_tag('tag3'), b'san')
    t.commit(allow_pointless=True, message='second commit', rev_id=b'second-revid')
    t2 = WorkingTree.open('branch2')
    t2.commit(allow_pointless=True, message='commit in second')
    b1.tags.set_tag('tag4', b'second-revid')
    self.run_bzr('merge -d branch2 branch1')
    self.assertEqual(b2.tags.lookup_tag('tag4'), b'second-revid')
    self.run_bzr('push -d branch1 branch3')
    b3 = Branch.open('branch3')
    self.assertEqual(b3.tags.lookup_tag('tag1'), b'first-revid')