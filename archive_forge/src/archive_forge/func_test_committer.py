import os
from breezy import branchbuilder, errors, log, osutils, tests
from breezy.tests import features, test_log
def test_committer(self):
    self.prepare_tree()
    self.assertLogRevnos(['-m', 'committer1'], ['1'])
    self.assertLogRevnos(['-m', 'committer2'], ['2'])
    self.assertLogRevnos(['-m', 'committer'], ['2', '1'])
    self.assertLogRevnos(['-m', 'committer1', '-m', 'committer2'], ['2', '1'])
    self.assertLogRevnos(['--match-committer', 'committer1'], ['1'])
    self.assertLogRevnos(['--match-committer', 'committer2'], ['2'])
    self.assertLogRevnos(['--match-committer', 'committer'], ['2', '1'])
    self.assertLogRevnos(['--match-committer', 'committer1', '--match-committer', 'committer2'], ['2', '1'])