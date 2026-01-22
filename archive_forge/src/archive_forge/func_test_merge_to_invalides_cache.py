from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_to_invalides_cache(self):
    b1, revids = self.make_write_locked_branch_with_one_tag()
    b2 = b1.controldir.sprout('b2').open_branch()
    b2.tags.set_tag('two', revids[1])
    b2.tags.merge_to(b1.tags)
    self.assertEqual({'one': revids[0], 'two': revids[1]}, b1.tags.get_tag_dict())