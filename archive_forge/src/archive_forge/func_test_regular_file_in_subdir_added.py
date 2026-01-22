import os
from breezy.tests import TestCaseWithTransport
def test_regular_file_in_subdir_added(self):
    with self.tracker:
        self.build_tree_contents([('tree/foo/',), ('tree/foo/blah', 'bar')])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'foo', 'foo/blah'})