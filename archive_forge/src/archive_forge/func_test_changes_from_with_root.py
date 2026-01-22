from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_changes_from_with_root(self):
    """Ensure the include_root option does what's expected."""
    wt = self.make_branch_and_tree('.')
    delta = wt.changes_from(wt.basis_tree())
    self.assertEqual(len(delta.added), 0)
    delta = wt.changes_from(wt.basis_tree(), include_root=True)
    self.assertEqual(len(delta.added), 1)
    self.assertEqual(delta.added[0].path[1], '')