from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_reverse_tag_dict(self):
    b, [target_revid1, target_revid2] = self.make_branch_with_revision_tuple('b', 2)
    b.tags.set_tag('tag-name', target_revid1)
    b.tags.set_tag('other-name', target_revid2)
    b = branch.Branch.open('b')
    self.assertEqual(dict(b.tags.get_reverse_tag_dict()), {target_revid1: {'tag-name'}, target_revid2: {'other-name'}})