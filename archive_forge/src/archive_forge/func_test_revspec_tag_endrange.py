import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_revspec_tag_endrange(self):
    self.requireFeature(features.sed_feature)
    wt = self.make_branch_and_tree('.', format='dirstate-tags')
    wt.branch.tags.set_tag('tag1', b'null:')
    wt.branch.tags.set_tag('tag2', b'null:')
    self.complete(['brz', 'log', '-r', '3..tag', ':', 't'])
    self.assertCompletionEquals('tag1', 'tag2')
    self.complete(['brz', 'log', '-r', '"3..tag:t'])
    self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')
    self.complete(['brz', 'log', '-r', "'3..tag:t"])
    self.assertCompletionEquals('3..tag:tag1', '3..tag:tag2')