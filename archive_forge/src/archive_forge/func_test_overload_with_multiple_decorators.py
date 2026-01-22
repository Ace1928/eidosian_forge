from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_overload_with_multiple_decorators(self):
    self.flakes('\n            from typing import overload\n            dec = lambda f: f\n\n            @dec\n            @overload\n            def f(x: int) -> int:\n                pass\n\n            @dec\n            @overload\n            def f(x: str) -> str:\n                pass\n\n            @dec\n            def f(x): return x\n       ')