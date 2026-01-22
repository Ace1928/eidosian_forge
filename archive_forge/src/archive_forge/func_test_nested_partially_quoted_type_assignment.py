from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_nested_partially_quoted_type_assignment(self):
    self.flakes("\n        from queue import Queue\n        from typing import Callable\n\n        Func = Callable[['Queue[str]'], None]\n        ")