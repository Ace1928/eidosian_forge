from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_tags_initially_empty(self):
    b = self.make_branch('b')
    tags = b.tags.get_tag_dict()
    self.assertEqual(tags, {})