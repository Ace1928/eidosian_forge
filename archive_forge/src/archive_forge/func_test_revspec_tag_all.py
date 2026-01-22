import os
import subprocess
import sys
import breezy
from breezy import commands, osutils, tests
from breezy.plugins.bash_completion.bashcomp import *
from breezy.tests import features
def test_revspec_tag_all(self):
    self.requireFeature(features.sed_feature)
    wt = self.make_branch_and_tree('.', format='dirstate-tags')
    wt.branch.tags.set_tag('tag1', b'null:')
    wt.branch.tags.set_tag('tag2', b'null:')
    wt.branch.tags.set_tag('3tag', b'null:')
    self.complete(['brz', 'log', '-r', 'tag', ':'])
    self.assertCompletionEquals('tag1', 'tag2', '3tag')