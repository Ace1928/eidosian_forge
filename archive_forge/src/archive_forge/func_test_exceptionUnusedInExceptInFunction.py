from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_exceptionUnusedInExceptInFunction(self):
    self.flakes('\n        def download_review():\n            try: pass\n            except Exception as e: pass\n        ', m.UnusedVariable)