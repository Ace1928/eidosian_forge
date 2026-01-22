import unittest
from traits.api import (
def test_arg_check_decorator_trailing_comma(self):
    with self.assertRaises(TraitError):
        ArgCheckDecoratorTrailingComma(tc=self)