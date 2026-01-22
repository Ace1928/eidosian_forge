import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_modified_directory(self):
    """Test --directory option"""
    tree = self.make_branch_and_tree('a')
    self.build_tree(['a/README'])
    tree.add('README')
    tree.commit('r1')
    self.build_tree_contents([('a/README', b'changed\n')])
    out, err = self.run_bzr(['modified', '--directory=a'])
    self.assertEqual('README\n', out)