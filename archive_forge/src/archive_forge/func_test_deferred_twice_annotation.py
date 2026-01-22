from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_deferred_twice_annotation(self):
    self.flakes('\n            from queue import Queue\n            from typing import Optional\n\n\n            def f() -> "Optional[\'Queue[str]\']":\n                return None\n        ')