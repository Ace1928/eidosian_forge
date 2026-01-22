import os
import re
from breezy import tests, workingtree
from breezy.diff import DiffTree
from breezy.diff import format_registry as diff_format_registry
from breezy.tests import features
def test_binary_diff_remove(self):
    tree = self.make_branch_and_tree('.')
    self.build_tree_contents([('a', b'\x00' * 20)])
    tree.add(['a'])
    tree.commit('add binary file')
    os.unlink('a')
    output = self.run_bzr('diff', retcode=1)
    self.assertEqual("=== removed file 'a'\nBinary files old/a and new/a differ\n", output[0])