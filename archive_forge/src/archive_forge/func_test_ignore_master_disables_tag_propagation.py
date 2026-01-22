from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_ignore_master_disables_tag_propagation(self):
    """merge_to(child, ignore_master=True) does not merge tags to the
        master.
        """
    master = self.make_branch('master')
    other = self.make_branch('other')
    other.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    other.tags.merge_to(child.tags, ignore_master=True)
    self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
    self.assertRaises(errors.NoSuchTag, master.tags.lookup_tag, 'foo')