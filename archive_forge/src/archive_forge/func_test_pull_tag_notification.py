import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_tag_notification(self):
    """pulling tags with conflicts will change the exit code"""
    from_tree = self.make_branch_and_tree('from')
    from_tree.branch.tags.set_tag('mytag', b'somerevid')
    to_tree = self.make_branch_and_tree('to')
    out = self.run_bzr(['pull', '-d', 'to', 'from'])
    self.assertEqual(out, ('1 tag(s) updated.\n', ''))