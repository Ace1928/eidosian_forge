import unittest
from subunit.progress_model import ProgressModel
def test_set_width_absolute_preserves_pos(self):
    progress = ProgressModel()
    progress.advance()
    progress.set_width(2)
    self.assertProgressSummary(1, 2, progress)