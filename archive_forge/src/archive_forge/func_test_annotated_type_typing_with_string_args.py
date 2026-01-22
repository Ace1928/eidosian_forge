from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_annotated_type_typing_with_string_args(self):
    self.flakes("\n        from typing import Annotated\n\n        def f(x: Annotated[int, '> 0']) -> None:\n            return None\n        ")