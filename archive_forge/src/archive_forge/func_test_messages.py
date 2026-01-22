from breezy import conflicts, tests, workingtree
from breezy.tests import features, script
def test_messages(self):
    """Conflict messages involving non-ascii paths are displayed okay"""
    make_tree_with_conflicts(self, 'branch', prefix='§')
    out, err = self.run_bzr(['conflicts', '-d', 'branch'], encoding=self.encoding)
    self.assertEqual(out, 'Text conflict in §_other_file\nPath conflict: §dir3 / §dir2\nText conflict in §file\n')
    self.assertEqual(err, '')