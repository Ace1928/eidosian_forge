from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_lookup_tag(self):
    self.tags.set_tag('foo', b'revid1')
    self.assertEqual(b'revid1', self.tags.lookup_tag('foo'))
    self.assertRaises(errors.NoSuchTag, self.tags.lookup_tag, 'bar')