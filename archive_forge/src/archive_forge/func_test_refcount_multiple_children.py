from ... import errors, multiparent, tests
from .. import groupcompress, versionedfile
def test_refcount_multiple_children(self):
    vf = self.make_three_vf()
    gen = versionedfile._MPDiffGenerator(vf, [(b'two',), (b'three',)])
    needed_keys, refcount = gen._find_needed_keys()
    self.assertEqual(sorted([(b'one',), (b'two',), (b'three',)]), sorted(needed_keys))
    self.assertEqual({(b'one',): 2, (b'two',): 1}, refcount)
    self.assertEqual([(b'one',)], sorted(gen.present_parents))