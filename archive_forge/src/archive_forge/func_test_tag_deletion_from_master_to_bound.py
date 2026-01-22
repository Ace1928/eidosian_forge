from breezy import controldir, errors
from breezy.tag import DisabledTags, MemoryTags
from breezy.tests import TestCase, TestCaseWithTransport
def test_tag_deletion_from_master_to_bound(self):
    master = self.make_branch('master')
    master.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    master.tags.delete_tag('foo')
    self.knownFailure('tag deletion does not propagate: https://bugs.launchpad.net/bzr/+bug/138802')
    self.assertRaises(errors.NoSuchTag, child.tags.lookup_tag, 'foo')