from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_set_tag_invalides_cache(self):
    b, revids = self.make_write_locked_branch_with_one_tag()
    b.tags.set_tag('one', revids[1])
    self.assertEqual({'one': revids[1]}, b.tags.get_tag_dict())