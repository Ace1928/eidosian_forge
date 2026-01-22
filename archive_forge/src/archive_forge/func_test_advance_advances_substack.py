import unittest
from subunit.progress_model import ProgressModel
def test_advance_advances_substack(self):
    progress = ProgressModel()
    progress.adjust_width(3)
    progress.advance()
    progress.push()
    progress.adjust_width(1)
    progress.advance()
    self.assertProgressSummary(2, 3, progress)