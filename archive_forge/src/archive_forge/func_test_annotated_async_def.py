from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_annotated_async_def(self):
    self.flakes('\n        class c: pass\n        async def func(c: c) -> None: pass\n        ')