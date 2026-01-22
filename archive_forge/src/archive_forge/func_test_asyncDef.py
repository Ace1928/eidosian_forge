from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skip, skipIf
def test_asyncDef(self):
    self.flakes('\n        async def bar():\n            return 42\n        ')