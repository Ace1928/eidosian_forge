from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_tag_copied_by_initial_checkout(self):
    master = self.make_branch('master')
    master.tags.set_tag('foo', b'rev-1')
    co_tree = master.create_checkout('checkout')
    self.assertEqual(b'rev-1', co_tree.branch.tags.lookup_tag('foo'))