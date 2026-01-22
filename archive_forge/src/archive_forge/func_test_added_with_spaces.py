import os
from breezy.branch import Branch
from breezy.tests import TestCaseWithTransport
def test_added_with_spaces(self):
    """Test that 'added' command reports added files with spaces in their names quoted"""
    self._test_added('a filename with spaces', '"a filename with spaces"\n')