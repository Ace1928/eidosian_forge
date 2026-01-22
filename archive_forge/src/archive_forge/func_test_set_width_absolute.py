import unittest
from subunit.progress_model import ProgressModel
def test_set_width_absolute(self):
    progress = ProgressModel()
    progress.set_width(10)
    self.assertProgressSummary(0, 10, progress)