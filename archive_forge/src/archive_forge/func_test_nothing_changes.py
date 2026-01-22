import os
from breezy.tests import TestCaseWithTransport
def test_nothing_changes(self):
    with self.tracker:
        self.assertFalse(self.tracker.is_dirty())