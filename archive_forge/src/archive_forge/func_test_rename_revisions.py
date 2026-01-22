from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_rename_revisions(self):
    self.tags.set_tag('foo', b'revid1')
    self.assertEqual({'foo': b'revid1'}, self.tags.get_tag_dict())
    self.tags.rename_revisions({b'revid1': b'revid2'})
    self.assertEqual({'foo': b'revid2'}, self.tags.get_tag_dict())