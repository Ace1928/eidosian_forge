import os
from breezy.tests import TestCaseWithTransport
def test_regular_file_added(self):
    with self.tracker:
        self.build_tree_contents([('tree/foo', 'bar')])
        self.assertTrue(self.tracker.is_dirty())
        self.assertEqual(self.tracker.relpaths(), {'foo'})