from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_exceptionUsedInExcept(self):
    self.flakes('\n        try: pass\n        except Exception as e: e\n        ')
    self.flakes('\n        def download_review():\n            try: pass\n            except Exception as e: e\n        ')