import ast
import os
import re
import sys
import breezy.branch
from breezy import osutils
from breezy.tests import TestCase, TestSkipped, features
def test_branch_WorkingTree(self):
    """Test that the number of uses of working_tree in branch is stable."""
    occurences = self.find_occurences('WorkingTree', self.source_file_name(breezy.branch))
    self.assertEqual(0, occurences)