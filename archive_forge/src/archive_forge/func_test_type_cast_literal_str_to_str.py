from sys import version_info
from pyflakes import messages as m
from pyflakes.test.harness import TestCase, skipIf
def test_type_cast_literal_str_to_str(self):
    self.flakes("\n        from typing import cast\n\n        a_string = cast(str, 'Optional[int]')\n        ")