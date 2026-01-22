import doctest
import os
from testtools import matchers
from breezy import (branch, controldir, merge_directive, osutils, tests,
from breezy.bzr import conflicts
from breezy.tests import scenarios, script
def run_merge_then_revert(args, retcode=None, working_dir='a'):
    self.run_bzr(['merge', '../b', '-r', 'last:1..last:1'] + args, retcode=retcode, working_dir=working_dir)
    if retcode != 3:
        a_tree.revert(backups=False)