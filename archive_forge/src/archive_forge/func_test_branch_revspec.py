import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_branch_revspec(self):
    foo = self.make_branch_and_tree('foo')
    bar = self.make_branch_and_tree('bar')
    self.build_tree(['foo/foo.txt', 'bar/bar.txt'])
    foo.add('foo.txt')
    bar.add('bar.txt')
    foo.commit(message='foo')
    bar.commit(message='bar')
    self.run_bzr('log -r branch:../bar', working_dir='foo')
    self.assertEqual([bar.branch.get_rev_id(1)], [r.rev.revision_id for r in self.get_captured_revisions()])