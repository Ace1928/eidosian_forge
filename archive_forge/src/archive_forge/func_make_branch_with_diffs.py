import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def make_branch_with_diffs(self):
    level0 = self.make_branch_and_tree('level0')
    self.build_tree(['level0/file1', 'level0/file2'])
    level0.add('file1')
    level0.add('file2')
    self.wt_commit(level0, 'in branch level0')
    level1 = level0.controldir.sprout('level1').open_workingtree()
    self.build_tree_contents([('level1/file2', b'hello\n')])
    self.wt_commit(level1, 'in branch level1')
    level0.merge_from_branch(level1.branch)
    self.wt_commit(level0, 'merge branch level1')