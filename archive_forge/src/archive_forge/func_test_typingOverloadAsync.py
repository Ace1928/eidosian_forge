from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_typingOverloadAsync(self):
    """Allow intentional redefinitions via @typing.overload (async)"""
    self.flakes('\n        from typing import overload\n\n        @overload\n        async def f(s: None) -> None:\n            pass\n\n        @overload\n        async def f(s: int) -> int:\n            pass\n\n        async def f(s):\n            return s\n        ')