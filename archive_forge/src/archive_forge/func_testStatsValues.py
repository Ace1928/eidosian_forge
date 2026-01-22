import os.path
from breezy.iterablefile import IterableFile
from breezy.patches import (NO_NL, AppliedPatches, BinaryFiles, BinaryPatch,
from breezy.tests import TestCase, TestCaseWithTransport
def testStatsValues(self):
    """Test the added, removed and hunks values for stats_values."""
    patch = parse_patch(self.datafile('diff'))
    self.assertEqual((299, 407, 48), patch.stats_values())