import os
import sys
from breezy import (branch, debug, osutils, tests, uncommit, urlutils,
from breezy.bzr import remote
from breezy.directory_service import directories
from breezy.tests import fixtures, script
def test_pull_tag_overwrite(self):
    """pulling tags with --overwrite only reports changed tags."""
    from_tree = self.make_branch_and_tree('from')
    from_tree.branch.tags.set_tag('mytag', b'somerevid')
    to_tree = self.make_branch_and_tree('to')
    to_tree.branch.tags.set_tag('mytag', b'somerevid')
    out = self.run_bzr(['pull', '--overwrite', '-d', 'to', 'from'])
    self.assertEqual(out, ('No revisions or tags to pull.\n', ''))