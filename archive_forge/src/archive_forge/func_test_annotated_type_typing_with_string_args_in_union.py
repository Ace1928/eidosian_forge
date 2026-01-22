from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_annotated_type_typing_with_string_args_in_union(self):
    self.flakes("\n        from typing import Annotated, Union\n\n        def f(x: Union[Annotated['int', '>0'], 'integer']) -> None:\n            return None\n        ", m.UndefinedName)