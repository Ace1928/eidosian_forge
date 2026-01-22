import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_log_with_tags(self):
    tree = self.make_linear_branch(format='dirstate-tags')
    branch = tree.branch
    branch.tags.set_tag('tag1', branch.get_rev_id(1))
    branch.tags.set_tag('tag1.1', branch.get_rev_id(1))
    branch.tags.set_tag('tag3', branch.last_revision())
    log = self.run_bzr('log -r-1')[0]
    self.assertTrue('tags: tag3' in log)
    log = self.run_bzr('log -r1')[0]
    self.assertContainsRe(log, 'tags: (tag1, tag1\\.1|tag1\\.1, tag1)')