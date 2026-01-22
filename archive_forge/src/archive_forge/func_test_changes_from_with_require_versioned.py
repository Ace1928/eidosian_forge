from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_changes_from_with_require_versioned(self):
    """Ensure the require_versioned option does what's expected."""
    wt = self.make_branch_and_tree('.')
    self.build_tree(['known_file', 'unknown_file'])
    wt.add('known_file')
    self.assertRaises(errors.PathsNotVersionedError, wt.changes_from, wt.basis_tree(), wt, specific_files=['known_file', 'unknown_file'], require_versioned=True)
    delta = wt.changes_from(wt.basis_tree(), specific_files=['known_file', 'unknown_file'], require_versioned=False)
    self.assertEqual(len(delta.added), 1)