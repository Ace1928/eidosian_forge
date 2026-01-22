import os
from breezy.tests import TestCaseWithTransport
def test_ancestry_with_location(self):
    """Tests 'ancestry' command with a specified location."""
    self._build_branches()
    self._check_ancestry('A')