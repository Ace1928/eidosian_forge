from breezy import branch as _mod_branch
from breezy import errors, lockable_files, lockdir, tag
from breezy.branch import Branch
from breezy.bzr import branch as bzrbranch
from breezy.bzr import bzrdir
from breezy.tests import TestCaseWithTransport, script
from breezy.workingtree import WorkingTree
def test_list_tags(self):
    tree1 = self.make_branch_and_tree('branch1')
    tree1.commit(allow_pointless=True, message='revision 1', rev_id=b'revid-1', timestamp=10)
    tree1.commit(allow_pointless=True, message='revision 2', rev_id=b'revid-2', timestamp=15)
    b1 = tree1.branch
    b1.tags.set_tag('tag1バ', b'revid-2')
    b1.tags.set_tag('tag10バ', b'missing')
    b1.tags.set_tag('tag2バ', b'revid-1')
    out, err = self.run_bzr_raw('tags -d branch1', encoding='utf-8')
    self.assertEqual(err, b'')
    self.assertContainsRe(out, ('^tag1バ  *2\ntag2バ  *1\n' + 'tag10バ *\\?\n').encode('utf-8'))
    out, err = self.run_bzr_raw('tags --sort=alpha -d branch1', encoding='utf-8')
    self.assertEqual(err, b'')
    self.assertContainsRe(out, ('^tag10バ  *\\?\ntag1バ  *2\n' + 'tag2バ *1\n').encode('utf-8'))
    out, err = self.run_bzr_raw('tags --sort=alpha --show-ids -d branch1', encoding='utf-8')
    self.assertEqual(err, b'')
    self.assertContainsRe(out, ('^tag10バ  *missing\n' + 'tag1バ  *revid-2\ntag2バ *revid-1\n').encode('utf-8'))
    out, err = self.run_bzr_raw('tags --sort=time -d branch1', encoding='utf-8')
    self.assertEqual(err, b'')
    self.assertContainsRe(out, ('^tag2バ  *1\ntag1バ  *2\n' + 'tag10バ *\\?\n').encode('utf-8'))
    out, err = self.run_bzr_raw('tags --sort=time --show-ids -d branch1', encoding='utf-8')
    self.assertEqual(err, b'')
    self.assertContainsRe(out, ('^tag2バ  *revid-1\n' + 'tag1バ  *revid-2\ntag10バ *missing\n').encode('utf-8'))
    tree2 = tree1.controldir.sprout('branch2').open_workingtree()
    tree1.commit(allow_pointless=True, message='revision 3 in branch1', rev_id=b'revid-3a')
    tree2.commit(allow_pointless=True, message='revision 3 in branch2', rev_id=b'revid-3b')
    b2 = tree2.branch
    b2.tags.set_tag('tagD', b'revid-3b')
    self.run_bzr('merge -d branch1 branch2')
    tree1.commit('merge', rev_id=b'revid-4')
    out, err = self.run_bzr('tags -d branch1', encoding='utf-8')
    self.assertEqual(err, '')
    self.assertContainsRe(out, 'tagD  *2\\.1\\.1\\n')
    out, err = self.run_bzr('tags -d branch2', encoding='utf-8')
    self.assertEqual(err, '')
    self.assertContainsRe(out, 'tagD  *3\\n')