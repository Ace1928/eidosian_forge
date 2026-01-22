from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_typingExtensionsOverload(self):
    """Allow intentional redefinitions via @typing_extensions.overload"""
    self.flakes('\n        import typing_extensions\n        from typing_extensions import overload\n\n        @overload\n        def f(s: None) -> None:\n            pass\n\n        @overload\n        def f(s: int) -> int:\n            pass\n\n        def f(s):\n            return s\n\n        @typing_extensions.overload\n        def g(s: None) -> None:\n            pass\n\n        @typing_extensions.overload\n        def g(s: int) -> int:\n            pass\n\n        def g(s):\n            return s\n        ')