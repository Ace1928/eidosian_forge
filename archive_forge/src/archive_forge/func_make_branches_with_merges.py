import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def make_branches_with_merges(self):
    level0 = self.make_branch_and_tree('level0')
    self.wt_commit(level0, 'in branch level0')
    level1 = level0.controldir.sprout('level1').open_workingtree()
    self.wt_commit(level1, 'in branch level1')
    level2 = level1.controldir.sprout('level2').open_workingtree()
    self.wt_commit(level2, 'in branch level2')
    level1.merge_from_branch(level2.branch)
    self.wt_commit(level1, 'merge branch level2')
    level0.merge_from_branch(level1.branch)
    self.wt_commit(level0, 'merge branch level1')