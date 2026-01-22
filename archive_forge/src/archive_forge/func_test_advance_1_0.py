import unittest
from subunit.progress_model import ProgressModel
def test_advance_1_0(self):
    progress = ProgressModel()
    progress.advance()
    self.assertProgressSummary(1, 0, progress)