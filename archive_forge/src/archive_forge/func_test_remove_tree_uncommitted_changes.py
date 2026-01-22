import os
from breezy import shelf
from breezy.tests import TestCaseWithTransport
def test_remove_tree_uncommitted_changes(self):
    self.build_tree(['branch1/bar'])
    self.tree.add('bar')
    output = self.run_bzr_error(['Working tree .* has uncommitted changes'], 'remove-tree branch1', retcode=3)