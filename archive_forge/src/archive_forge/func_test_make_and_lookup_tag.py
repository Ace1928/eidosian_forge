from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_make_and_lookup_tag(self):
    b, [target_revid1, target_revid2] = self.make_branch_with_revision_tuple('b', 2)
    b.tags.set_tag('tag-name', target_revid1)
    b.tags.set_tag('other-name', target_revid2)
    b = branch.Branch.open('b')
    self.assertEqual(b.tags.get_tag_dict(), {'tag-name': target_revid1, 'other-name': target_revid2})
    result = b.tags.lookup_tag('tag-name')
    self.assertEqual(result, target_revid1)
    self.assertTrue(b.tags.has_tag('tag-name'))
    self.assertFalse(b.tags.has_tag('imaginary'))