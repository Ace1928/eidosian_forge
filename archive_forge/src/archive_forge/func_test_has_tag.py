from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_has_tag(self):
    self.tags.set_tag('foo', b'revid1')
    self.assertTrue(self.tags.has_tag('foo'))
    self.assertFalse(self.tags.has_tag('bar'))