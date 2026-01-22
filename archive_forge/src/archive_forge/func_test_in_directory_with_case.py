from typing import List, Tuple
from breezy import errors, revision
from breezy.tests import TestCase, TestCaseWithTransport
from breezy.tree import (FileTimestampUnavailable, InterTree,
def test_in_directory_with_case(self):
    tree = self.make_branch_and_tree('tree')
    self.build_tree(['tree/a/', 'tree/a/b'])
    tree.add(['a', 'a/b'])
    self.assertEqual('a/b', get_canonical_path(tree, 'a/b', lambda x: x.lower()))
    self.assertEqual('a/b', get_canonical_path(tree, 'A/B', lambda x: x.lower()))
    self.assertEqual('a/b', get_canonical_path(tree, 'A/b', lambda x: x.lower()))
    self.assertEqual('a/C', get_canonical_path(tree, 'A/C', lambda x: x.lower()))