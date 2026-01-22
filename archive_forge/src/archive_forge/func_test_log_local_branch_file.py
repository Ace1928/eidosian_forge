import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_local_branch_file(self):
    """We should be able to log files in local treeless branches"""
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/file'])
    tree.add('file')
    tree.commit('revision 1')
    tree.controldir.destroy_workingtree()
    self.run_bzr('log tree/file')