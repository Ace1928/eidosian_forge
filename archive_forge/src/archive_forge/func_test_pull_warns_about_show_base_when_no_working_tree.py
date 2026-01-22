import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_warns_about_show_base_when_no_working_tree(self):
    """--show-base is useless if there's no working tree

        see https://bugs.launchpad.net/bzr/+bug/1022160"""
    self.make_branch('from')
    self.make_branch('to')
    out = self.run_bzr(['pull', '-d', 'to', 'from', '--show-base'])
    self.assertEqual(out, ('No revisions or tags to pull.\n', 'No working tree, ignoring --show-base\n'))