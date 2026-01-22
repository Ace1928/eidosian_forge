from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_tag_methods_raise(self):
    b = self.make_branch('b')
    self.assertRaises(errors.TagsNotSupported, b.tags.set_tag, 'foo', 'bar')
    self.assertRaises(errors.TagsNotSupported, b.tags.lookup_tag, 'foo')
    self.assertRaises(errors.TagsNotSupported, b.tags.set_tag, 'foo', 'bar')
    self.assertRaises(errors.TagsNotSupported, b.tags.delete_tag, 'foo')