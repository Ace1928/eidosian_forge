import time
from .... import tests
from ..helpers import kind_to_mode
from . import FastimportFeature
def test_modify_revert(self):
    handler, branch = self.get_handler()
    handler.process(self.file_command_iter())
    branch.lock_read()
    self.addCleanup(branch.unlock)
    rev_d = branch.last_revision()
    rev_a, rev_c = branch.repository.get_parent_map([rev_d])[rev_d]
    rev_b = branch.repository.get_parent_map([rev_c])[rev_c][0]
    rtree_a, rtree_b, rtree_c, rtree_d = branch.repository.revision_trees([rev_a, rev_b, rev_c, rev_d])
    self.assertEqual(rev_a, rtree_a.get_file_revision('foo'))
    self.assertEqual(rev_b, rtree_b.get_file_revision('foo'))
    self.assertEqual(rev_c, rtree_c.get_file_revision('foo'))
    self.assertEqual(rev_c, rtree_d.get_file_revision('foo'))