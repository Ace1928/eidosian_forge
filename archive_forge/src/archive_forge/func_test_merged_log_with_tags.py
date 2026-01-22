import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_merged_log_with_tags(self):
    branch1_tree = self.make_linear_branch('branch1', format='dirstate-tags')
    branch1 = branch1_tree.branch
    branch2_tree = branch1_tree.controldir.sprout('branch2').open_workingtree()
    branch1_tree.commit(message='foobar', allow_pointless=True)
    branch1.tags.set_tag('tag1', branch1.last_revision())
    self.run_bzr('merge ../branch1', working_dir='branch2')
    branch2_tree.commit(message='merge branch 1')
    log = self.run_bzr('log -n0 -r-1', working_dir='branch2')[0]
    self.assertContainsRe(log, '    tags: tag1')
    log = self.run_bzr('log -n0 -r3.1.1', working_dir='branch2')[0]
    self.assertContainsRe(log, 'tags: tag1')