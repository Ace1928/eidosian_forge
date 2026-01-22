import unittest
from subunit.progress_model import ProgressModel
def test_pop_restores_progress(self):
    progress = ProgressModel()
    progress.adjust_width(3)
    progress.advance()
    progress.push()
    progress.adjust_width(1)
    progress.advance()
    progress.pop()
    self.assertProgressSummary(1, 3, progress)