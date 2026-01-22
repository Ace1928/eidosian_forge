from breezy import branch, controldir, errors, tests
from breezy.bzr import branch as bzrbranch
from breezy.tests import per_branch
def test_merge_to_overwrite_conflict_in_master(self):
    """merge_to(child, overwrite=True) overwrites any conflicting tags in
        the master.
        """
    master = self.make_branch('master')
    other = self.make_branch('other')
    other.tags.set_tag('foo', b'rev-1')
    child = self.make_branch('child')
    child.bind(master)
    child.update()
    master.tags.set_tag('foo', b'rev-2')
    tag_updates, tag_conflicts = other.tags.merge_to(child.tags, overwrite=True)
    self.assertEqual(b'rev-1', child.tags.lookup_tag('foo'))
    self.assertEqual(b'rev-1', master.tags.lookup_tag('foo'))
    self.assertEqual({'foo': b'rev-1'}, tag_updates)
    self.assertLength(0, tag_conflicts)